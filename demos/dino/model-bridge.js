/**
 * Zesyde Dino Showcase - Model Bridge
 * High-performance WebSocket and HTTP client connecting game telemetry
 * directly to the Zesyde 150M-class ONNX CPU inference server.
 */

class ZesydeModelBridge {
  constructor(options = {}) {
    this.wsUrl = options.wsUrl || ((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    this.apiUrl = options.apiUrl || "/api/game-decision";
    this.ws = null;
    this.pendingResolvers = new Map();
    this.reqId = 0;
    this.isWsReady = false;
    this.initWebSocket();
  }

  initWebSocket() {
    try {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => {
        this.isWsReady = true;
        console.log("[ZesydeBridge] WebSocket connection established.");
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Oldest pending resolver gets the response
          if (this.pendingResolvers.size > 0) {
            const firstKey = this.pendingResolvers.keys().next().value;
            const resolver = this.pendingResolvers.get(firstKey);
            this.pendingResolvers.delete(firstKey);
            resolver(data);
          }
        } catch (err) {
          console.error("[ZesydeBridge] WS parse error:", err);
        }
      };

      this.ws.onerror = (err) => {
        this.isWsReady = false;
      };

      this.ws.onclose = () => {
        this.isWsReady = false;
      };
    } catch (e) {
      this.isWsReady = false;
    }
  }

  /**
   * Request a real-time tactical decision from the Zesyde model.
   * @param {Object} params - Telemetry packet { distance, speed, obstacleType, elevation, language }
   * @returns {Promise<Object|null>} Decision packet with action, confidence, latency_ms
   */
  async decide(params) {
    const payload = {
      distance: params.distance ?? 500,
      speed: params.speed ?? 6.0,
      obstacle_type: params.obstacleType || params.obstacle_type || "none",
      elevation: params.elevation || "low",
      time_to_impact: params.speed > 0 ? (params.distance / (params.speed * 60)) : 0.5,
      language: params.language || "en",
    };

    // If WebSocket is open and ready, use it for lowest latency
    if (this.isWsReady && this.ws && this.ws.readyState === WebSocket.OPEN) {
      return new Promise((resolve) => {
        const id = ++this.reqId;
        const timer = setTimeout(() => {
          this.pendingResolvers.delete(id);
          // Fall back to HTTP fetch on WS timeout
          this.fetchHttp(payload).then(resolve);
        }, 120);

        this.pendingResolvers.set(id, (res) => {
          clearTimeout(timer);
          resolve(res);
        });

        this.ws.send(JSON.stringify(payload));
      });
    }

    // Otherwise use HTTP POST
    return this.fetchHttp(payload);
  }

  async fetchHttp(payload) {
    try {
      const res = await fetch(this.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  }
}

window.ZesydeModelBridge = ZesydeModelBridge;
