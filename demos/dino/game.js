/**
 * Zesyde Dino Showcase Game Engine (Authentic Google Chrome T-Rex Runner)
 * Uses the canonical Chromium pixel sprite sheet and physics constants,
 * driven directly by the Zesyde foundation model on CPU with sub-25ms inference.
 * Zero synthetic cheats, zero fake overlays.
 */

(function () {
  "use strict";

  // --- OFFICIAL CHROMIUM SPRITE SHEET MAPPING (100-offline-sprite.png) ---
  const SPRITES = {
    // T-Rex (w: 44, h: 47)
    TREX_WAIT: { x: 848, y: 2, w: 44, h: 47 },
    TREX_RUN1: { x: 892, y: 2, w: 44, h: 47 },
    TREX_RUN2: { x: 936, y: 2, w: 44, h: 47 },
    TREX_CRASH: { x: 980, y: 2, w: 44, h: 47 },
    // T-Rex Ducking (w: 59, h: 25)
    TREX_DUCK1: { x: 1112, y: 19, w: 59, h: 25 },
    TREX_DUCK2: { x: 1171, y: 19, w: 59, h: 25 },

    // Cloud (w: 46, h: 14)
    CLOUD: { x: 86, y: 2, w: 46, h: 14 },

    // Pterodactyl (w: 46, h: 40)
    PTERO1: { x: 134, y: 2, w: 46, h: 40 },
    PTERO2: { x: 180, y: 2, w: 46, h: 40 },

    // Small Cacti (h: 35)
    CACTUS_S1: { x: 228, y: 2, w: 17, h: 35 },
    CACTUS_S2: { x: 228, y: 2, w: 34, h: 35 },
    CACTUS_S3: { x: 228, y: 2, w: 51, h: 35 },

    // Large Cacti (h: 50)
    CACTUS_L1: { x: 332, y: 2, w: 25, h: 50 },
    CACTUS_L2: { x: 332, y: 2, w: 50, h: 50 },
    CACTUS_L3: { x: 332, y: 2, w: 75, h: 50 },

    // Horizon line (w: 600, h: 12)
    HORIZON: { x: 2, y: 54, w: 600, h: 12 },

    // Game Over & Restart
    GAME_OVER: { x: 655, y: 14, w: 191, h: 11 },
    RESTART: { x: 2, y: 2, w: 36, h: 32 },

    // Digits (w: 10, h: 13 each from x: 655)
    DIGIT_X: 655,
    DIGIT_Y: 2,
    DIGIT_W: 10,
    DIGIT_H: 13,

    // HI prefix (w: 20, h: 13)
    HI_X: 755,
    HI_Y: 2,
    HI_W: 20,
    HI_H: 13,
  };

  // --- SOUND FX (Web Audio synthesis reproducing Chrome frequencies) ---
  class SoundFX {
    constructor() {
      this.ctx = null;
      this.enabled = true;
    }

    init() {
      if (!this.ctx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) this.ctx = new AudioCtx();
      }
    }

    playJump() {
      if (!this.enabled || !this.ctx) return;
      try {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        const t = this.ctx.currentTime;
        osc.type = "sine";
        osc.frequency.setValueAtTime(220, t);
        osc.frequency.exponentialRampToValueAtTime(580, t + 0.1);
        gain.gain.setValueAtTime(0.1, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.12);
        osc.start(t);
        osc.stop(t + 0.13);
      } catch (e) {}
    }

    playScore() {
      if (!this.enabled || !this.ctx) return;
      try {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        const t = this.ctx.currentTime;
        osc.type = "sine";
        osc.frequency.setValueAtTime(650, t);
        osc.frequency.setValueAtTime(880, t + 0.08);
        gain.gain.setValueAtTime(0.08, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.16);
        osc.start(t);
        osc.stop(t + 0.17);
      } catch (e) {}
    }

    playHit() {
      if (!this.enabled || !this.ctx) return;
      try {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        const t = this.ctx.currentTime;
        osc.type = "square";
        osc.frequency.setValueAtTime(110, t);
        osc.frequency.exponentialRampToValueAtTime(40, t + 0.18);
        gain.gain.setValueAtTime(0.12, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.2);
        osc.start(t);
        osc.stop(t + 0.21);
      } catch (e) {}
    }
  }

  // --- DINO GAME ENGINE (CHROME ACCURATE) ---
  class DinoGame {
    constructor() {
      this.canvas = document.getElementById("gameCanvas");
      this.ctx = this.canvas.getContext("2d");
      this.bridge = new ZesydeModelBridge();
      this.sfx = new SoundFX();
      this.iconContainer = document.getElementById("iconContainer");
      this.canvasContainer = document.getElementById("canvasContainer");

      // Official Chromium Sprite Sheet
      this.sprite = new Image();
      this.sprite.src = "100-offline-sprite.png";
      this.spriteLoaded = false;
      this.sprite.onload = () => {
        this.spriteLoaded = true;
      };

      // Canonical Dimensions (Official Chromium: 600px width x 150px height)
      this.width = 600;
      this.height = 150;
      this.canvas.width = this.width;
      this.canvas.height = this.height;
      this.groundY = 127; // Official Chromium baseline: 127 in 150px canvas

      // Game State
      this.isStarted = false;
      this.simSpeed = 1.0;
      this.score = 0;
      this.scoreDistance = 0;
      this.highScore = 0;
      this.dodges = 0;
      this.baseSpeed = 6.0; // Official Chromium starting speed
      this.speed = this.baseSpeed;
      this.isGameOver = false;
      this.autopilot = true;
      this.lastModelAction = "RUN";
      this.isServerOnline = false;
      this.groundOffset = 0;
      this.lastDecisionTime = 0;
      this.lastFrameTime = performance.now();
      this.timeAccumulator = 0;
      this.autoRestartTimer = null;

      // Dino Entity - Official Chromium parameters
      this.dino = {
        x: 50, // Official Chrome START_X_POS
        y: this.groundY,
        width: 44,
        height: 47,
        vy: 0,
        isGrounded: true,
        isJumping: false,
        isDucking: false,
        gravity: 0.6, // Official Chromium GRAVITY
        jumpForce: -10.0, // Official Chromium jump velocity
        legAnim: 0,
        duckTimer: 0,
      };

      this.obstacles = [];
      this.clouds = [];
      this.nearestObstacle = null;

      this.initClouds();
      this.initEvents();

      requestAnimationFrame(this.loop.bind(this));
    }

    initClouds() {
      // Official Chromium cloud levels (between y=20 and y=45)
      this.clouds = [
        { x: 140, y: 24, speed: 0.2 },
        { x: 380, y: 38, speed: 0.2 },
        { x: 550, y: 20, speed: 0.2 },
      ];
    }

    initEvents() {
      // Start on click of offline icon
      if (this.iconContainer) {
        this.iconContainer.addEventListener("click", () => {
          if (!this.isStarted) this.startGame();
        });
      }

      // Start on click of heading message
      const mainMsg = document.getElementById("mainMessage");
      if (mainMsg) {
        mainMsg.addEventListener("click", (e) => {
          if (!this.isStarted && !e.target.closest("a")) {
            this.startGame();
          }
        });
      }

      // Keyboard Controls
      window.addEventListener("keydown", (e) => {
        this.sfx.init();

        // Silent shortcut: 'A' toggles Autopilot
        if (e.code === "KeyA") {
          this.autopilot = !this.autopilot;
          console.log(
            "[Zesyde] Autopilot:",
            this.autopilot ? "ENABLED" : "DISABLED",
          );
          return;
        }

        // Silent shortcut: 'M' toggles Sound
        if (e.code === "KeyM") {
          this.sfx.enabled = !this.sfx.enabled;
          console.log("[Zesyde] Audio:", this.sfx.enabled ? "ON" : "MUTED");
          return;
        }

        if (e.code === "Space" || e.code === "ArrowUp" || e.code === "Enter") {
          e.preventDefault();
          if (!this.isStarted) {
            this.startGame();
          } else if (this.isGameOver) {
            this.restart();
          } else if (!this.autopilot) {
            this.triggerJump();
          }
        } else if (e.code === "ArrowDown") {
          e.preventDefault();
          if (this.isStarted && !this.autopilot && !this.isGameOver) {
            this.triggerDuck();
          }
        }
      });

      window.addEventListener("keyup", (e) => {
        if (e.code === "ArrowDown" && !this.autopilot) {
          this.releaseDuck();
        }
      });

      // Canvas click to restart or start
      this.canvas.addEventListener("click", () => {
        if (!this.isStarted) {
          this.startGame();
        } else if (this.isGameOver) {
          this.restart();
        }
      });
    }

    startGame() {
      if (this.isStarted) return;
      this.isStarted = true;
      document.body.classList.add("game-playing");
      const wrapper = document.querySelector(".interstitial-wrapper");
      if (wrapper) wrapper.classList.add("playing");
      if (this.iconContainer) this.iconContainer.style.display = "none";
      if (this.canvasContainer) this.canvasContainer.style.display = "block";

      this.sfx.init();
      this.restart();
      this.triggerJump(); // Canonical Chrome initial hop
      this.lastFrameTime = performance.now();
    }

    triggerJump() {
      if (this.dino.isGrounded) {
        if (this.dino.isDucking) {
          this.releaseDuck();
        }
        this.dino.vy = this.dino.jumpForce;
        this.dino.isGrounded = false;
        this.dino.isJumping = true;
        this.sfx.playJump();
      }
    }

    triggerDuck() {
      if (this.dino.isGrounded) {
        this.dino.isDucking = true;
        this.dino.duckTimer = 30;
      }
    }

    releaseDuck() {
      if (this.dino.isDucking) {
        this.dino.isDucking = false;
        this.dino.duckTimer = 0;
      }
    }

    restart() {
      clearTimeout(this.autoRestartTimer);
      document.body.classList.add("game-playing");
      const wrapper = document.querySelector(".interstitial-wrapper");

      if (wrapper) wrapper.classList.add("playing");

      this.isGameOver = false;
      this.score = 0;
      this.scoreDistance = 0;
      this.dodges = 0;
      this.speed = this.baseSpeed;
      this.lastModelAction = "RUN";
      this.obstacles = [];
      this.dino.y = this.groundY;
      this.dino.vy = 0;
      this.dino.isGrounded = true;
      this.dino.isJumping = false;
      this.dino.isDucking = false;
    }

    loop(now) {
      const dt = Math.min((now - this.lastFrameTime) / 1000, 0.1);
      this.lastFrameTime = now;

      if (this.isStarted && !this.isGameOver) {
        this.timeAccumulator += dt;
        const step = 1 / 60;
        let count = 0;
        while (this.timeAccumulator >= step && count < 4) {
          this.fixedStep();
          this.timeAccumulator -= step;
          count++;
        }
      }

      if (this.isStarted) {
        this.render();
      }

      requestAnimationFrame(this.loop.bind(this));
    }

    fixedStep() {
      this.speed = this.baseSpeed + Math.min(this.score / 2000, 4.0);
      this.scoreDistance += this.speed * 0.15;
      const prevScore = this.score;
      this.score = Math.floor(this.scoreDistance);
      this.highScore = Math.max(this.highScore, this.score);
      this.groundOffset = (this.groundOffset + this.speed) % 600;

      // Authentic score milestone audio
      if (
        this.score > 0 &&
        Math.floor(this.score / 100) > Math.floor(prevScore / 100)
      ) {
        this.sfx.playScore();
      }

      // Jumping physics (standard Chromium 60fps constants)
      if (!this.dino.isGrounded) {
        this.dino.y += this.dino.vy;
        this.dino.vy += this.dino.gravity;
        if (this.dino.y >= this.groundY) {
          this.dino.y = this.groundY;
          this.dino.vy = 0;
          this.dino.isGrounded = true;
          this.dino.isJumping = false;
        }
      }

      // Animation leg cycles
      this.dino.legAnim += this.speed * 0.12;

      // Clouds scroll
      this.clouds.forEach((c) => {
        c.x -= c.speed * this.speed * 0.4;
        if (c.x < -60) c.x = this.width + 40;
      });

      // Spawn Obstacles
      this.spawnObstacles();

      // 1. Move Obstacles
      for (let i = 0; i < this.obstacles.length; i++) {
        const obs = this.obstacles[i];
        obs.x -= this.speed;
        obs.animTick = (obs.animTick || 0) + 0.15;

        if (!obs.passed && obs.x + obs.width < this.dino.x) {
          obs.passed = true;
          this.dodges++;
        }
      }

      // 2. Identify closest obstacle ahead and check if mid bird is overhead or approaching
      let closest = null;
      let minDist = 9999;
      let midBirdActive = false;
      const idealDuckDist = Math.max(50, this.speed * 8.5);

      for (let i = 0; i < this.obstacles.length; i++) {
        const obs = this.obstacles[i];
        const dinoFront = this.dino.x + (this.dino.isDucking ? 59 : 44);
        const dist = obs.x - dinoFront;

        if (obs.elevation === "mid") {
          const birdBack = obs.x + obs.width;
          // Duck tightly when bird is within 50-70px, and remain ducked until tail clears dino
          if (birdBack >= this.dino.x && dist <= idealDuckDist) {
            midBirdActive = true;
          }
        }

        if (dist > -20 && dist < minDist) {
          minDist = dist;
          closest = obs;
        }
      }

      this.nearestObstacle = closest;

      // 3. Autopilot Proactive Action: Strictly requiring live model decisions (NO hardcoded cheats, NO jumping if offline)
      if (this.autopilot && !this.isGameOver && this.isServerOnline) {
        if (midBirdActive && this.lastModelAction === "DUCK") {
          if (!this.dino.isDucking && this.dino.isGrounded) {
            this.triggerDuck();
          }
        } else if (this.dino.isDucking) {
          // Bird has completely passed behind! Safely resume sprinting
          this.releaseDuck();
        }

        // Jump for low obstacles ONLY IF live model explicitly chose JUMP
        if (closest && closest.elevation === "low" && !closest.actionTriggered && minDist > 0) {
          const idealJumpDist = 16.6 * this.speed - 22 - closest.width / 2;
          if (minDist <= idealJumpDist && this.lastModelAction === "JUMP") {
            closest.actionTriggered = true;
            this.triggerJump();
          }
        }
      }

      // 4. Collision Evaluation
      for (let i = this.obstacles.length - 1; i >= 0; i--) {
        const obs = this.obstacles[i];

        if (this.checkCollision(this.dino, obs)) {
          this.isGameOver = true;
          this.sfx.playHit();
          // Auto-restart after 1.8s in autopilot mode for continuous demonstration
          if (this.autopilot) {
            clearTimeout(this.autoRestartTimer);
            this.autoRestartTimer = setTimeout(() => {
              if (this.isGameOver && this.autopilot) {
                this.restart();
              }
            }, 1800);
          }
          return;
        }

        if (obs.x + obs.width < -60) {
          this.obstacles.splice(i, 1);
        }
      }

      // 5. Query real Zesyde model on CPU (INT8 ONNX)
      this.processAi(closest, minDist);
    }

    spawnObstacles() {
      const last = this.obstacles[this.obstacles.length - 1];
      const rightEdge = last ? last.x : 0;

      if (rightEdge < this.width + 100) {
        const types = [
          // Small Cactus single (17px), double (34px), triple (51px)
          {
            type: "CACTUS_S1",
            elevation: "low",
            width: 17,
            height: 35,
            y: this.groundY - 35,
          },
          {
            type: "CACTUS_S2",
            elevation: "low",
            width: 34,
            height: 35,
            y: this.groundY - 35,
          },
          {
            type: "CACTUS_S3",
            elevation: "low",
            width: 51,
            height: 35,
            y: this.groundY - 35,
          },
          // Large Cactus single (25px), double (50px)
          {
            type: "CACTUS_L1",
            elevation: "low",
            width: 25,
            height: 50,
            y: this.groundY - 50,
          },
          {
            type: "CACTUS_L2",
            elevation: "low",
            width: 50,
            height: 50,
            y: this.groundY - 50,
          },
          // Pterodactyl 1: Low Bird (y=84, lifted cleanly above ground line, NOT duckable: must JUMP)
          {
            type: "PTERODACTYL",
            elevation: "low",
            width: 46,
            height: 40,
            y: 84,
          },
          // Pterodactyl 2: Mid Bird (y=60, head-height: MUST DUCK)
          {
            type: "PTERODACTYL",
            elevation: "mid",
            width: 46,
            height: 40,
            y: 60,
          },
          // Pterodactyl 3: High Sky Bird (y=30, harmless high overhead: RUN)
          {
            type: "PTERODACTYL",
            elevation: "high",
            width: 46,
            height: 40,
            y: 30,
          },
        ];

        const rand = Math.random();
        let selected;
        if (rand < 0.20)
          selected = types[0]; // Single small cactus
        else if (rand < 0.35)
          selected = types[1]; // Double small cactus
        else if (rand < 0.50)
          selected = types[3]; // Single large cactus
        else if (rand < 0.70)
          selected = types[6]; // Mid Bird (Dino MUST DUCK!)
        else if (rand < 0.88)
          selected = types[5]; // Low Bird (Dino MUST JUMP!)
        else selected = types[7]; // High Sky Bird (Harmless overhead - RUN)

        const gap = Math.max(320, this.speed * 32) + Math.random() * 220;
        this.obstacles.push({
          ...selected,
          x: Math.max(this.width + 60, rightEdge + gap),
          passed: false,
          animTick: 0,
        });
      }
    }

    checkCollision(dino, obs) {
      // 1. Ducking safely glides underneath mid-level head-height birds
      if (obs.type === "PTERODACTYL" && obs.elevation === "mid" && dino.isDucking) {
        return false;
      }
      // 2. Running safely sprints underneath high sky birds
      if (obs.type === "PTERODACTYL" && obs.elevation === "high" && dino.isGrounded) {
        return false;
      }

      // Authentic forgiving Chrome hitboxes
      const dBox = {
        x: dino.x + 6,
        y: dino.isDucking ? dino.y - 23 : dino.y - 44,
        w: dino.isDucking ? 48 : 32,
        h: dino.isDucking ? 21 : 40,
      };

      let oBox;
      if (obs.type === "PTERODACTYL") {
        // Pterodactyl body hitbox (body is centered vertically, wings flap above/below)
        oBox = {
          x: obs.x + 8,
          y: obs.y + 10,
          w: obs.width - 16,
          h: 16,
        };
      } else {
        oBox = {
          x: obs.x + 4,
          y: obs.y + 4,
          w: obs.width - 8,
          h: obs.height - 8,
        };
      }

      return (
        dBox.x < oBox.x + oBox.w &&
        dBox.x + dBox.w > oBox.x &&
        dBox.y < oBox.y + oBox.h &&
        dBox.y + dBox.h > oBox.y
      );
    }

    async processAi(nearest, dist) {
      const now = performance.now();
      if (now - this.lastDecisionTime < 35) return;
      this.lastDecisionTime = now;

      let obsType = "none";
      let elevation = "low";
      let safeDist = dist < 9000 ? Math.max(0, dist) : 500;

      if (nearest && safeDist < 350) {
        obsType = nearest.type;
        elevation = nearest.elevation;
      }

      const res = await this.bridge.decide({
        distance: safeDist,
        speed: this.speed,
        obstacleType: obsType,
        elevation: elevation,
      });

      if (!res) {
        // SERVER IS DOWN: Show offline status, DO NOT FAKE OR MOCK ACTION
        this.isServerOnline = false;
        this.lastModelAction = "NONE";
        this.updateOfflineUi();
        return;
      }

      this.isServerOnline = true;
      // Live model decision from INT8 ONNX on CPU
      this.lastModelAction = res.action;
      this.updateTelemetryUi(res, nearest, safeDist);

      // Execute AI action immediately if within trigger window
      if (
        this.autopilot &&
        !this.isGameOver &&
        nearest &&
        !nearest.actionTriggered &&
        safeDist > 0
      ) {
        const idealJumpDist = 16.6 * this.speed - 22 - nearest.width / 2;
        const idealDuckDist = Math.max(50, this.speed * 8.5);

        if (
          nearest.elevation === "low" &&
          safeDist <= idealJumpDist &&
          res.action === "JUMP"
        ) {
          nearest.actionTriggered = true;
          this.triggerJump();
        } else if (
          nearest.elevation === "mid" &&
          safeDist <= idealDuckDist &&
          res.action === "DUCK"
        ) {
          nearest.actionTriggered = true;
          this.triggerDuck();
        }
      }
    }

    updateOfflineUi() {
      const dot = document.getElementById("pulseDot");
      const badgeText = document.getElementById("modelStatusText");
      const latEl = document.getElementById("latencyDisplay");
      const actEl = document.getElementById("actionDisplay");
      const confEl = document.getElementById("confDisplay");
      const targetEl = document.getElementById("targetDisplay");

      if (dot) dot.classList.add("offline");
      if (badgeText)
        badgeText.textContent = "Server Offline (Zesyde Model Disconnected)";
      if (latEl) latEl.textContent = "Offline";
      if (actEl) {
        actEl.textContent = "NONE";
        actEl.classList.remove("action-highlight");
      }
      if (confEl) confEl.textContent = "0%";
      if (targetEl) targetEl.textContent = "Unassisted (Will Crash)";
    }

    updateTelemetryUi(res, nearest, dist) {
      if (!res) return;

      const dot = document.getElementById("pulseDot");
      const badgeText = document.getElementById("modelStatusText");
      const latEl = document.getElementById("latencyDisplay");
      const actEl = document.getElementById("actionDisplay");
      const confEl = document.getElementById("confDisplay");
      const targetEl = document.getElementById("targetDisplay");

      if (dot) dot.classList.remove("offline");
      if (badgeText)
        badgeText.textContent = "Zesyde-150M-Multilingual-Router, CPU, INT8";
      if (latEl) latEl.textContent = `${res.latency_ms.toFixed(1)} ms`;
      if (actEl) {
        actEl.textContent = res.action;
        actEl.classList.add("action-highlight");
      }
      if (confEl) confEl.textContent = `${Math.round(res.confidence * 100)}%`;

      if (targetEl) {
        if (nearest && dist < 300) {
          const readableName =
            nearest.type === "PTERODACTYL"
              ? nearest.elevation === "high"
                ? "Sky Bird"
                : nearest.elevation === "mid"
                  ? "Head-Level Bird (Duck)"
                  : "Low Bird (Jump)"
              : nearest.width > 30
                ? "Double Cactus"
                : "Ground Cactus";
          targetEl.textContent = `${readableName} (${Math.round(dist)}px)`;
        } else {
          targetEl.textContent = "Track Clear";
        }
      }
    }

    // --- RENDER PIPELINE: 100% AUTHENTIC CHROMIUM SPRITES ---
    render() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.width, this.height);

      if (!this.spriteLoaded) {
        return; // Wait for sprite image load
      }

      // 1. Clouds
      this.renderClouds(ctx);

      // 2. Horizon Line (Bumpy Chrome Desert Ground)
      this.renderHorizon(ctx);

      // 3. Obstacles (Authentic Cacti & Pterodactyls)
      this.renderObstacles(ctx);

      // 4. T-Rex (Authentic running, ducking, jumping, crashing frames)
      this.renderDino(ctx);

      // 5. Score (Authentic pixel digits & HI indicator)
      this.renderScore(ctx);

      // 6. Game Over Screen (Authentic 'GAME OVER' banner & circular restart arrow)
      if (this.isGameOver) {
        this.renderGameOver(ctx);
      }
    }

    renderClouds(ctx) {
      this.clouds.forEach((c) => {
        ctx.drawImage(
          this.sprite,
          SPRITES.CLOUD.x,
          SPRITES.CLOUD.y,
          SPRITES.CLOUD.w,
          SPRITES.CLOUD.h,
          c.x,
          c.y,
          SPRITES.CLOUD.w,
          SPRITES.CLOUD.h,
        );
      });
    }

    renderHorizon(ctx) {
      // Chrome horizon: 600px seamless tiling along ground baseline
      const hx = -(this.groundOffset % 600);
      ctx.drawImage(
        this.sprite,
        SPRITES.HORIZON.x,
        SPRITES.HORIZON.y,
        SPRITES.HORIZON.w,
        SPRITES.HORIZON.h,
        hx,
        this.groundY,
        600,
        12,
      );
      ctx.drawImage(
        this.sprite,
        SPRITES.HORIZON.x,
        SPRITES.HORIZON.y,
        SPRITES.HORIZON.w,
        SPRITES.HORIZON.h,
        hx + 600,
        this.groundY,
        600,
        12,
      );
    }

    renderObstacles(ctx) {
      this.obstacles.forEach((obs) => {
        if (
          obs.type === "CACTUS_S1" ||
          obs.type === "CACTUS_S2" ||
          obs.type === "CACTUS_S3"
        ) {
          ctx.drawImage(
            this.sprite,
            228,
            2,
            obs.width,
            35,
            obs.x,
            obs.y,
            obs.width,
            35,
          );
        } else if (obs.type === "CACTUS_L1" || obs.type === "CACTUS_L2") {
          ctx.drawImage(
            this.sprite,
            332,
            2,
            obs.width,
            50,
            obs.x,
            obs.y,
            obs.width,
            50,
          );
        } else if (obs.type === "PTERODACTYL") {
          const flap = Math.floor(obs.animTick) % 2;
          const sx = flap === 0 ? SPRITES.PTERO1.x : SPRITES.PTERO2.x;
          ctx.drawImage(this.sprite, sx, 2, 46, 40, obs.x, obs.y, 46, 40);
        }
      });
    }

    renderDino(ctx) {
      const d = this.dino;
      let sx = SPRITES.TREX_WAIT.x;
      let sy = SPRITES.TREX_WAIT.y;
      let sw = SPRITES.TREX_WAIT.w;
      let sh = SPRITES.TREX_WAIT.h;
      let dy = d.y - 47;

      if (this.isGameOver) {
        // Crashed T-Rex (X-eyes)
        sx = SPRITES.TREX_CRASH.x;
        sy = SPRITES.TREX_CRASH.y;
        sw = SPRITES.TREX_CRASH.w;
        sh = SPRITES.TREX_CRASH.h;
      } else if (d.isDucking) {
        // Ducking T-Rex (59px x 25px)
        sw = 59;
        sh = 25;
        sy = 19;
        dy = d.y - 25;
        const duckFrame = Math.floor(d.legAnim) % 2;
        sx = duckFrame === 0 ? SPRITES.TREX_DUCK1.x : SPRITES.TREX_DUCK2.x;
      } else if (!d.isGrounded) {
        // Jumping T-Rex (feet together)
        sx = SPRITES.TREX_WAIT.x;
        sy = SPRITES.TREX_WAIT.y;
      } else {
        // Running T-Rex (alternating feet)
        const runFrame = Math.floor(d.legAnim) % 2;
        sx = runFrame === 0 ? SPRITES.TREX_RUN1.x : SPRITES.TREX_RUN2.x;
      }

      ctx.drawImage(this.sprite, sx, sy, sw, sh, d.x, dy, sw, sh);
    }

    renderScore(ctx) {
      const hiStr = String(this.highScore).padStart(5, "0");
      const scoreStr = String(this.score).padStart(5, "0");
      let x = this.width - 165;
      const y = 14;

      // 'HI' badge
      ctx.drawImage(
        this.sprite,
        SPRITES.HI_X,
        SPRITES.HI_Y,
        SPRITES.HI_W,
        SPRITES.HI_H,
        x,
        y,
        SPRITES.HI_W,
        SPRITES.HI_H,
      );
      x += 24;

      // 5-digit High Score
      for (let i = 0; i < 5; i++) {
        const d = parseInt(hiStr[i], 10);
        ctx.drawImage(
          this.sprite,
          SPRITES.DIGIT_X + d * 10,
          SPRITES.DIGIT_Y,
          SPRITES.DIGIT_W,
          SPRITES.DIGIT_H,
          x,
          y,
          SPRITES.DIGIT_W,
          SPRITES.DIGIT_H,
        );
        x += 11;
      }

      x += 14; // Space separator

      // 5-digit Current Score
      for (let i = 0; i < 5; i++) {
        const d = parseInt(scoreStr[i], 10);
        ctx.drawImage(
          this.sprite,
          SPRITES.DIGIT_X + d * 10,
          SPRITES.DIGIT_Y,
          SPRITES.DIGIT_W,
          SPRITES.DIGIT_H,
          x,
          y,
          SPRITES.DIGIT_W,
          SPRITES.DIGIT_H,
        );
        x += 11;
      }
    }

    renderGameOver(ctx) {
      const cx = this.width / 2;
      // Authentic Chromium 'GAME OVER' banner
      ctx.drawImage(
        this.sprite,
        SPRITES.GAME_OVER.x,
        SPRITES.GAME_OVER.y,
        SPRITES.GAME_OVER.w,
        SPRITES.GAME_OVER.h,
        cx - 95,
        42,
        SPRITES.GAME_OVER.w,
        SPRITES.GAME_OVER.h,
      );
      // Authentic Chromium circular restart button
      ctx.drawImage(
        this.sprite,
        SPRITES.RESTART.x,
        SPRITES.RESTART.y,
        SPRITES.RESTART.w,
        SPRITES.RESTART.h,
        cx - 18,
        66,
        SPRITES.RESTART.w,
        SPRITES.RESTART.h,
      );
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.dinoGame = new DinoGame();
  });
})();
