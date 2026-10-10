"""
Comprehensive 300-Sample Handpicked Benchmark for Zesyde.

Evaluates Zesyde across 3 realistic categories:
1. Legitimate Emails (100 samples): Transactional receipts, work memos, shipping, account notices, multi-lang ham.
2. Spam & Phishing Emails (100 samples): Phishing, 419 fraud, crypto giveaways, fake DHL/FedEx, extortion, multi-lang spam.
3. General Facts & Reasoning (100 samples): Science, geography, history, technology, logic, multi-lang QA.

Computes:
- Accuracy overall & per-category
- Spam/Ham Precision, Recall, and F1
- Escalation Head metrics (trigger rate, mean escalate prob)
- Latency (mean, p50, p95)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zesyde.lang_router import LanguageRouter


def get_handpicked_dataset() -> List[Dict[str, Any]]:
    """Generates the curated 300-sample handpicked evaluation suite."""
    examples = []

    # =========================================================================
    # CATEGORY 1: LEGITIMATE EMAILS (100 SAMPLES)
    # =========================================================================
    
    # E-Commerce & Transactional Receipts (25)
    receipts = [
        ("From: orders@amazon.com\nSubject: Order confirmation #114-928192\nHi David, thank you for your order. We will notify you once items ship.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: receipts@uber.com\nSubject: Your trip with Uber on Friday\nTotal: $24.50 charged to Visa ...1042. Pickup: 5th Ave. Dropoff: JFK Terminal 4.", "Categorize this email:", ["Spam", "Legitimate"], 1),
        ("From: service@paypal.com\nSubject: Receipt for your payment to Spotify AB\nYou sent a payment of $10.99 USD to Spotify AB. Transaction ID: 89X72819.", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: auto-confirm@apple.com\nSubject: Your receipt from Apple\nInvoice ID: INV-981248. iCloud+ 200GB storage plan: $2.99. Billed to Apple ID.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: billing@stripe.com\nSubject: Invoice #1029 paid for Acme Corp\nYour payment for Acme Pro monthly subscription was successfully processed.", "What kind of email is this?", ["Spam", "Legitimate"], 1),
        ("From: automated@airbnb.com\nSubject: Reservation confirmed for Tokyo\nHost: Kenji. Check-in: Nov 14, 2026. Address: Shibuya City 2-11. Total: $480.", "Identify email category:", ["Spam", "Legitimate"], 1),
        ("From: donotreply@steamgames.com\nSubject: Thank you for your Steam purchase!\nItem: Cyberpunk 2077. Total: $29.99. Wallet balance remaining: $14.20.", "Is this spam or ham?", ["Spam", "Legitimate"], 1),
        ("From: orders@doordash.com\nSubject: Order #7718 placed with Chipotle\nEstimated delivery: 7:15 PM - 7:35 PM. Your Dasher is Marcus.", "What is the nature of this email?", ["Spam", "Legitimate"], 1),
        ("From: payments@google.com\nSubject: Google Play order receipt from Oct 10\nOrder number: GPA.3392-8172. Item: YouTube Premium individual membership.", "What is this message?", ["Spam", "Legitimate"], 1),
        ("From: notifications@target.com\nSubject: Your Target Drive Up order is ready\nStore: Northgate. Pickup spot: Stalls 1-6. Please open your app upon arrival.", "Email classification:", ["Spam", "Legitimate"], 1),
        ("From: support@bestbuy.com\nSubject: Best Buy order BBY01-882910 is ready for pickup\nItem: Samsung 2TB SSD 990 Pro. Please bring a valid government ID.", "Is this email spam or valid?", ["Spam", "Legitimate"], 1),
        ("From: travel@delta.com\nSubject: Your flight receipt: NYC to LHR\nConfirmation: H9XK2P. Passenger: Sarah Connor. Seat: 14B. Terminal 4 departure.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: receipts@lyft.com\nSubject: Your ride on Thursday morning\nRide cost: $18.20. Driver: Carlos (Toyota Prius). Tip included: $3.00.", "Is this email legitimate or spam?", ["Spam", "Legitimate"], 1),
        ("From: notify@zomato.com\nSubject: Order delivered from Burger King\nWe hope you enjoyed your meal! Rate your delivery partner Ramesh.", "Email classification:", ["Spam", "Legitimate"], 1),
        ("From: billing@digitalocean.com\nSubject: Monthly invoice for team workspace\nDroplet compute charges for September 2026: $48.00. Payment method: Mastercard.", "What is this communication?", ["Spam", "Legitimate"], 1),
        ("From: accounts@github.com\nSubject: Payment receipt for GitHub Copilot Business\nSeats: 5. Amount: $95.00 billed to corporate American Express card.", "Is this communication legitimate?", ["Spam", "Legitimate"], 1),
        ("From: info@booking.com\nSubject: Your booking at Hotel Arts Barcelona is confirmed\nBooking number: 4928172. Check-in: Dec 20. Free cancellation until Dec 18.", "Categorize this email:", ["Spam", "Legitimate"], 1),
        ("From: orders@walmart.com\nSubject: Tracking details for Walmart order #881290\nShipped via OnTrac. Tracking: 940011189922. Estimated delivery: Monday.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: payments@etsy.com\nSubject: Order confirmation for Handcrafted Leather Journal\nShop: ArtisansWorkshop. Ship to: 742 Evergreen Terrace.", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: support@cloudflare.com\nSubject: Cloudflare subscription renewal receipt\nPlan: Pro plan ($20/month) renewed for domain example.org.", "Identify email category:", ["Spam", "Legitimate"], 1),
        ("From: updates@sephora.com\nSubject: Beauty Insider points update and order confirmation\nOrder #S99214. Points earned: 85. Standard 3-day shipping selected.", "Email type:", ["Spam", "Legitimate"], 1),
        ("From: automated@costco.com\nSubject: Costco.com order #829188 submitted\nItems: Kirkland Signature paper towels (2-pack). Delivery window: Oct 14.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: billing@notion.so\nSubject: Receipt for Notion Plus Workspace\nWorkspace: Personal Wiki. Billing cycle: Annual ($96.00).", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: support@patreon.com\nSubject: Your monthly membership receipt\nYou supported 'Hardcore History' with $5.00 for October 2026.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: receipts@starbucks.com\nSubject: Reload receipt for Starbucks Card #9281\nReload amount: $25.00. Auto-reload triggered from Chase checking account.", "Is this spam or ham?", ["Spam", "Legitimate"], 1),
    ]

    # Shipping, Delivery & Tracking (25)
    shipping = [
        ("From: tracking@fedex.com\nSubject: FedEx Shipment 789123456789: Package delivered\nDelivered to front porch at 2:15 PM. Signature not required.", "What is this notification?", ["Spam", "Legitimate"], 1),
        ("From: pkginfo@ups.com\nSubject: UPS Update: Tracking #1Z9999999999999999\nScheduled delivery date changed to Thursday, Oct 12 by end of day.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: auto-reply@usps.com\nSubject: USPS Expected Delivery for Tracking 9400100000000000000000\nYour package has arrived at the regional sorting facility in Chicago.", "Is this email spam or valid?", ["Spam", "Legitimate"], 1),
        ("From: track@dhl.com\nSubject: DHL Express Waybill 4910291829: Customs clearance complete\nYour shipment has cleared customs and is out for local delivery in Munich.", "Identify email type:", ["Spam", "Legitimate"], 1),
        ("From: shipping@ikea.com\nSubject: Your IKEA delivery is scheduled for Saturday\nDelivery truck arrival window: 9:00 AM - 1:00 PM. Room of choice delivery.", "Is this legitimate?", ["Spam", "Legitimate"], 1),
        ("From: alerts@dpd.com\nSubject: DPD Parcel update: 1 hour delivery time slot\nDriver Sven will deliver your parcel between 11:20 and 12:20.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: notifications@royalmall.com\nSubject: Royal Mail: Item signed for and delivered\nItem reference: JG8291024GB. Handed to resident at front door.", "Categorize this email:", ["Spam", "Legitimate"], 1),
        ("From: deliveries@instacart.com\nSubject: Shopper has begun picking items for your order\nShopper Elena is at Whole Foods Market. Tap in app to approve replacements.", "Is this spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: tracking@purolator.com\nSubject: Purolator package on vehicle for delivery\nTracking: PIN 339102910. Destination: Toronto, ON.", "Classify this communication:", ["Spam", "Legitimate"], 1),
        ("From: status@gls-group.eu\nSubject: GLS tracking update: Parcel arrived at depot\nDepot: Utrecht 3500. Next step: Loaded onto delivery van.", "Is this spam or valid?", ["Spam", "Legitimate"], 1),
        ("From: update@blue-dart.com\nSubject: BlueDart Waybill 881928189 out for delivery\nLocation: Bangalore North. Delivery executive assigned: Rajesh.", "Identify email category:", ["Spam", "Legitimate"], 1),
        ("From: orders@zara.com\nSubject: Zara online order #99210 has been dispatched\nShipped via SEUR express. Expected delivery in 48 hours.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: support@asos.com\nSubject: Your ASOS parcel is on its way!\nTracking link: Hermes/Evri H092182. Thanks for shopping with ASOS.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: dispatch@hm.com\nSubject: H&M Order confirmation and delivery information\nOrder 7712910 has been packed at our distribution center.", "Is this spam or ham?", ["Spam", "Legitimate"], 1),
        ("From: shipping@nike.com\nSubject: Nike order 44102919 has shipped\nCarrier: FedEx. Tracking: 928102910281. Item: Air Jordan 1 Low Retro.", "Categorize this email:", ["Spam", "Legitimate"], 1),
        ("From: alerts@couriersplease.com.au\nSubject: CouriersPlease delivery notification\nConsignment #CP882910. Destination: Melbourne CBD.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: delivery@postnl.nl\nSubject: PostNL: Je pakket is onderweg\nBezorgtijd tussen 14:00 en 16:30. Bekijk actuele bezorgtijd in de PostNL app.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: tracking@chronopost.fr\nSubject: Votre colis Chronopost n° 99281920 arrive aujourd'hui\nLe livreur passera entre 10h15 et 11h15 à votre domicile.", "What kind of email is this?", ["Spam", "Legitimate"], 1),
        ("From: paket@dhl.de\nSubject: DHL Paketankündigung für Sendung 00340434\nIhre Sendung von Thalia Bücher wird voraussichtlich morgen zugestellt.", "Is this spam or valid?", ["Spam", "Legitimate"], 1),
        ("From: info@correos.es\nSubject: Correos España: Tu envío PQ8829182 está en reparto\nEl cartero intentará la entrega a lo largo del día de hoy.", "Identify email type:", ["Spam", "Legitimate"], 1),
        ("From: updates@delhivery.com\nSubject: Delhivery parcel tracking ID 1492810281\nStatus: Reached destination hub. Out for delivery in New Delhi.", "What is this notification?", ["Spam", "Legitimate"], 1),
        ("From: help@wayfair.com\nSubject: Wayfair: Large item delivery appointment confirmed\nItem: Dining table. Date: Oct 19. Team will call 30 mins before arrival.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: orders@chewy.com\nSubject: Chewy autoship order #99281 has shipped\nCarrier: FedEx Home Delivery. Tracking: 882910284719.", "Email classification:", ["Spam", "Legitimate"], 1),
        ("From: support@lululemon.com\nSubject: Your lululemon order is on the move\nTracking: 1Z8829102919. Shipped from our Columbus fulfillment center.", "Is this spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: updates@sephora.com\nSubject: Your beauty package has arrived!\nDelivered to parcel locker. Access code: 4920.", "Categorize this email:", ["Spam", "Legitimate"], 1),
    ]

    # Workplace & Collaboration (25)
    workplace = [
        ("From: notifications@github.com\nSubject: [Repo/api] Pull request #42 merged by alex\nPR title: 'Refactor auth middleware to support bearer tokens'. Branch main updated.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: jira@atlassian.net\nSubject: [JIRA] (PROJ-204) User profile modal unresponsive on mobile\nAssigned to: Maria Santos. Priority: High. Status changed to In Progress.", "Is this email spam or valid?", ["Spam", "Legitimate"], 1),
        ("From: notification@slack.com\nSubject: Unread mentions in #engineering-leads\nSarah mentioned you: '@john can you review the deploy runbook before 4 PM?'", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: calendar-notification@google.com\nSubject: Invitation: Q4 Architecture Review @ Thu Oct 15 2pm - 3pm\nOrganizer: David Kim. Video call: meet.google.com/abc-defg-hij.", "Email type:", ["Spam", "Legitimate"], 1),
        ("From: hr@internal-corp.com\nSubject: Annual open enrollment for healthcare benefits opens Monday\nPlease review health insurance plan tiers in Workday before Nov 1.", "Is this legitimate internal memo?", ["Spam", "Legitimate"], 1),
        ("From: alerts@pagerduty.com\nSubject: [RESOLVED] Incident #9921: High memory usage on prod-db-02\nTriggered at 03:14 UTC. Resolved by automated failover.", "Identify email category:", ["Spam", "Legitimate"], 1),
        ("From: devops@internal-corp.com\nSubject: Scheduled maintenance window: Saturday 02:00 - 04:00 UTC\nWe will be upgrading the staging Kubernetes cluster to v1.31.", "Is this communication spam?", ["Spam", "Legitimate"], 1),
        ("From: zoom-meetings@zoom.us\nSubject: Meeting summary and AI companion recap for Daily Standup\nAttendees: 8. Action items: John to push hotfix, Lisa to test staging.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: notifications@figma.com\nSubject: Elena commented on 'Mobile App v3 Designs'\n'Looks great! Can we adjust the corner radius on the primary CTA button?'", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: alerts@datadoghq.com\nSubject: Monitor [API 5xx error rate] recovered to OK\nCurrent metric value: 0.02% (below threshold of 1.0%).", "Email classification:", ["Spam", "Legitimate"], 1),
        ("From: support@gitlab.com\nSubject: Pipeline #88192 passed on branch feat/multi-token\nAll 24 test jobs succeeded in 3m 42s.", "Is this valid notification?", ["Spam", "Legitimate"], 1),
        ("From: notifications@asana.com\nSubject: Task due tomorrow: 'Submit quarterly marketing report'\nProject: 2026 Marketing Initiatives. Assigned to: You.", "What is this communication?", ["Spam", "Legitimate"], 1),
        ("From: team@trello.com\nSubject: New activity on board 'Sprint 44'\nCard 'Optimize database indices' was moved to Done by Marcus.", "Is this spam or ham?", ["Spam", "Legitimate"], 1),
        ("From: internal-payroll@company.org\nSubject: Your pay stub for period ending Oct 05 is available\nPlease log in to ADP Workforce Now to view your net pay statement.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: it-helpdesk@company.org\nSubject: Ticket #4491 closed: VPN access configuration\nYour request has been fulfilled. The configuration file was added to your profile.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: security@company.org\nSubject: Monthly phishing simulation test recap\nCongratulations! You correctly reported the simulated suspicious email last week.", "What kind of email is this?", ["Spam", "Legitimate"], 1),
        ("From: notifications@linear.app\nSubject: Issue ENG-982 assigned to you: 'Cache warming for search endpoint'\nPriority: Urgent. Cycle: Cycle 12.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: donotreply@docusign.net\nSubject: Completed: Mutual NDA with Quantum Leap Technologies\nAll parties have signed the document. A copy is attached for your records.", "Email type:", ["Spam", "Legitimate"], 1),
        ("From: alerts@sentry.io\nSubject: 1 new issue in production: TypeError: Cannot read property 'id'\nFirst seen 12 minutes ago in auth_service.py:144.", "Is this spam or valid?", ["Spam", "Legitimate"], 1),
        ("From: noreply@salesforce.com\nSubject: Lead assigned: Global Acme Enterprise\nNew high-value opportunity created by territory inbound router.", "Identify email category:", ["Spam", "Legitimate"], 1),
        ("From: community@stackoverflow.com\nSubject: Weekly newsletter: top questions in Python and Machine Learning\nCurated digest of highest-voted questions and accepted answers.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: alerts@aws.amazon.com\nSubject: AWS Budget Alert: Actual spend has exceeded 85% of monthly budget\nAccount ID: 991829102. Actual spend: $852.10 against $1,000 threshold.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: no-reply@zoominfo.com\nSubject: Weekly export report ready for download\n50 company profiles exported to CSV. Link active for 7 days in account dashboard.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: donotreply@workday.com\nSubject: Performance Review Cycle: Self-evaluation submitted\nYour manager will review your submission prior to calibration meeting.", "Is this spam or ham?", ["Spam", "Legitimate"], 1),
        ("From: feedback@lattice.com\nSubject: 1-on-1 agenda items updated by your manager\nMeeting time: Friday 11:00 AM. 3 topics queued for discussion.", "What is this communication?", ["Spam", "Legitimate"], 1),
    ]

    # Account Security & Multi-Language Ham (25)
    account_and_multilingual = [
        ("From: no-reply@accounts.google.com\nSubject: Security alert: New sign-in on Windows\nYour Google Account was accessed from Chrome on Windows in Chicago, IL. If this was you, no action needed.", "Is this email legitimate security notice?", ["Spam", "Legitimate"], 1),
        ("From: account-security-noreply@accountprotection.microsoft.com\nSubject: Microsoft account security code\nYour single-use code is 849201. If you did not request this, you can safely ignore this email.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: no-reply@chase.com\nSubject: Chase Account Alert: Direct deposit received\nA direct deposit of $3,250.00 from ACME CORP was posted to your checking account ...8812.", "What is this email?", ["Spam", "Legitimate"], 1),
        ("From: alerts@bankofamerica.com\nSubject: Your credit card statement is now available\nStatement balance: $1,420.15. Minimum payment due: $35.00 on Nov 02.", "Is this spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: secure@wellsfargo.com\nSubject: Wells Fargo Alert: ATM withdrawal confirmation\nAn ATM withdrawal of $100.00 was made on Oct 09 at 4:12 PM.", "Categorize this email:", ["Spam", "Legitimate"], 1),
        ("From: security@spotify.com\nSubject: Password changed successfully\nThe password for your Spotify account was changed on Friday. If you made this change, you're all set.", "Is this valid notice?", ["Spam", "Legitimate"], 1),
        ("From: notifications@discord.com\nSubject: New login from a new IP address\nLocation: Austin, Texas. Device: Discord Desktop App. Verify login with the button below.", "Email type:", ["Spam", "Legitimate"], 1),
        ("From: support@epicgames.com\nSubject: Two-factor authentication code\nYour Epic Games two-factor authentication code is 491028. Code expires in 10 minutes.", "Classify this email:", ["Spam", "Legitimate"], 1),
        ("From: noreply@steampowered.com\nSubject: Access from new web browser\nHere is the Steam Guard code you need to login: D49X2.", "Is this email legitimate?", ["Spam", "Legitimate"], 1),
        ("From: account@netflix.com\nSubject: A new device is using your account\nNetflix was accessed on a Roku TV in Atlanta, GA on Oct 08 at 8:44 PM.", "What kind of email is this?", ["Spam", "Legitimate"], 1),
        ("From: notify@uber.com\nSubject: Su código de verificación de Uber\nTu código de verificación de Uber es 3821. No compartas este código con nadie.", "Categoriza este correo:", ["Spam", "Legitimate"], 1),
        ("From: confirmation@renfe.com\nSubject: Billete confirmado: Madrid Atocha - Barcelona Sants\nLocalizador: K8291A. Coche 4, Plaza 12A. Salida 08:30 h.", "Es este correo legítimo o spam?", ["Spam", "Legitimate"], 1),
        ("From: servicioalcliente@mercadolibre.com\nSubject: ¡Tu compra está en camino!\nCompraste: Auriculares inalámbricos. El paquete llegará entre mañana y el viernes.", "Clasifica este mensaje:", ["Spam", "Legitimate"], 1),
        ("From: alertes@sncf-connect.com\nSubject: Confirmation de votre billet TGV INOUI Paris - Lyon\nDossier voyage: XN9928. Départ 14h28 Gare de Lyon.", "Quel est ce courriel?", ["Spam", "Legitimate"], 1),
        ("From: noreply@ameli.fr\nSubject: Vos remboursements de soins du mois de septembre\nUn nouveau relevé de prestations est disponible sur votre compte ameli sécurisé.", "Ce courriel est-il légitime ou spam?", ["Spam", "Legitimate"], 1),
        ("From: buchung@bahn.de\nSubject: Ihre Buchungsbestätigung für ICE 572 Berlin - Frankfurt\nAuftragsnummer: 8829104. Wagen 21, Sitzplatz 54 (Großraum, Fenster).", "Ist diese E-Mail legitim?", ["Spam", "Legitimate"], 1),
        ("From: kundenservice@telekom.de\nSubject: Ihre Telekom Festnetz Rechnung für September 2026\nRechnungsbetrag: 44,95 EUR. Der Betrag wird von Ihrem Konto abgebucht.", "Ist das Spam oder legitim?", ["Spam", "Legitimate"], 1),
        ("From: alerts@sbi.co.in\nSubject: SBI NetBanking: OTP for Funds Transfer\nYour OTP is 728190 for transfer of Rs 5000. Do not share OTP with bank staff.", "Classify this banking email:", ["Spam", "Legitimate"], 1),
        ("From: updates@irctc.co.in\nSubject: Electronic Reservation Slip (ERS) PNR 2819281029\nTrain: 12951 Mumbai Rajdhani. Passenger: Rajesh Sharma. Seat B2 34.", "Is this message spam or legitimate?", ["Spam", "Legitimate"], 1),
        ("From: receipts@swiggy.in\nSubject: Invoice for Swiggy Order #99182910\nDelivered from Punjab Grill. Total paid: Rs 840 via UPI. Thank you for ordering.", "What is this communication?", ["Spam", "Legitimate"], 1),
        ("From: service@paytm.com\nSubject: Fastag recharge successful of Rs. 1000\nVehicle: KA-01-MJ-4412. Toll balance updated successfully.", "Is this valid transaction email?", ["Spam", "Legitimate"], 1),
        ("From: alerts@icicibank.com\nSubject: Account Alert: Salary credit in account ...4421\nYour account has been credited with INR 1,25,000 on 30-Sep-2026.", "Email classification:", ["Spam", "Legitimate"], 1),
        ("From: confirm@docplanner.com\nSubject: Cita médica confirmada con el Dr. Martínez\nFecha: Lunes 16 de Octubre a las 11:30. Consulta en Clínica Central.", "Es este correo spam?", ["Spam", "Legitimate"], 1),
        ("From: no-reply@doctolib.fr\nSubject: Rappel de rendez-vous médical le 14 octobre à 10h00\nPraticien: Dr. Dupont, Médecin généraliste. Pensez à apporter votre carte Vitale.", "Classification de l'email:", ["Spam", "Legitimate"], 1),
        ("From: support@duolingo.com\nSubject: Don't lose your 150-day streak!\nTake 3 minutes today to practice Spanish and keep your streak alive.", "What is this email?", ["Spam", "Legitimate"], 1),
    ]

    for item in receipts + shipping + workplace + account_and_multilingual:
        examples.append({
            "category": "Legitimate Email",
            "context": item[0],
            "question": item[1],
            "options": item[2],
            "target": item[3]
        })

    # =========================================================================
    # CATEGORY 2: SPAM & PHISHING EMAILS (100 SAMPLES)
    # =========================================================================

    # Credential Harvesting & Account Phishing (25)
    credential_phishing = [
        ("From: security-alert@paypal-update-login-391.com\nSubject: URGENT: Your account has been restricted!\nWe detected unauthorized access. Click here immediately to verify your password and SSN: http://fake-login-phish.net\nYour account will be suspended within 24 hours.", "What is the category of this email?", ["Spam", "Legitimate"], 0),
        ("From: service@apple-id-verify-alert.top\nSubject: Your iCloud account is locked due to illegal activity\nTo unlock your account, click the link and provide your full credit card details and Apple ID password immediately.", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 0),
        ("From: netflix-billing@netflx-account-renew.xyz\nSubject: Final Notice: Payment declined on Netflix subscription\nYour membership is suspended. Update your card now at http://netflix-secure-billing-login.info to continue watching.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: admin@micro-soft-365-login-verify.com\nSubject: Microsoft Office 365: Password expires in 2 hours\nKeep your current password by verifying your credentials through our global server portal.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: security@chase-secure-online-banking-auth.co\nSubject: Unusual activity detected on your Chase debit card\nYour card has been temporarily frozen. Sign in here to enter your PIN and CVV to restore full access.", "Is this email legitimate or spam?", ["Spam", "Legitimate"], 0),
        ("From: verify@google-account-recovery-desk.site\nSubject: Critical security vulnerability on your Gmail\nForeign hackers in Russia logged into your email. Provide your 2FA recovery code to cancel their access.", "Identify email type:", ["Spam", "Legitimate"], 0),
        ("From: alerts@wellsfargo-verify-security-now.cc\nSubject: Wells Fargo: Mandatory identity re-verification required\nFederal banking regulations require you to re-upload your passport and debit card photos immediately.", "Is this message spam?", ["Spam", "Legitimate"], 0),
        ("From: support@meta-instagram-copyright-appeal.biz\nSubject: Copyright infringement on your Instagram account\nYour account will be permanently deleted in 48 hours. Submit an appeal with your password here.", "Classify this communication:", ["Spam", "Legitimate"], 0),
        ("From: billing@spotify-premium-renewal-failed.club\nSubject: Your Spotify Premium account was cancelled\nYour bank rejected the payment. Click here to submit a new credit card and get 6 months free.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: support@steam-community-free-wallet-giveaway.ru\nSubject: Claim your $50 Steam gift card from Gabe Newell\nLogin with your Steam credentials to claim your code directly into your inventory.", "Is this spam or ham?", ["Spam", "Legitimate"], 0),
        ("From: security@amazon-order-cancellation-desk.online\nSubject: Unauthorized order of Apple MacBook Pro $2,499.00\nIf you did not make this purchase, call 1-800-FAKE-NUM or enter your credit card on our portal to cancel.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: help@bankofamerica-customer-auth-relink.top\nSubject: Wire transfer of $4,850 is pending authorization\nDid you approve this wire to Nigeria? Click cancel now and enter your online banking credentials.", "Is this communication spam?", ["Spam", "Legitimate"], 0),
        ("From: admin@zoom-meeting-secure-join-391.net\nSubject: You have an urgent confidential video meeting waiting\nDownload the new ZoomWebPlugin.exe to join this room before it expires in 10 minutes.", "Identify email category:", ["Spam", "Legitimate"], 0),
        ("From: alerts@docu-sign-encrypted-contract-view.icu\nSubject: Important confidential legal document requires your electronic signature\nOpen the macro-enabled Word document attachment to sign.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: hr@internal-corp-benefits-login-phish.biz\nSubject: Urgent: Employee Q4 bonus structure announced\nDownload bonus_calculator.xls and enable macros to calculate your December bonus payout.", "Is this email legitimate or malicious spam?", ["Spam", "Legitimate"], 0),
        ("From: security@coinbase-wallet-reauth.org\nSubject: Coinbase: Your crypto wallet will be seized\nIRS compliance audit requires you to enter your 12-word seed phrase on our secure verification portal.", "Email classification:", ["Spam", "Legitimate"], 0),
        ("From: support@binance-security-kyc-update.com\nSubject: Binance Withdrawal disabled: Complete KYC level 3\nSend 0.05 BTC to our auditing wallet to unfreeze your balance.", "Is this spam or legitimate?", ["Spam", "Legitimate"], 0),
        ("From: verification@metamask-vault-restore.site\nSubject: MetaMask: Your Ethereum wallet requires migration\nMigrate to the new Ethereum V2 smart contract by syncing your private keys.", "What kind of email is this?", ["Spam", "Legitimate"], 0),
        ("From: accounts@twitter-x-verified-badge.info\nSubject: Congratulations! You are eligible for free blue checkmark\nEnter your Twitter username and current password to verify your identity.", "Is this spam?", ["Spam", "Legitimate"], 0),
        ("From: notify@linkedin-recruit-message-desk.xyz\nSubject: A Fortune 500 executive sent you a private job offer ($250k)\nLogin with your corporate email password to view the offer letter.", "Classify this message:", ["Spam", "Legitimate"], 0),
        ("From: support@facebook-security-infringement-case.me\nSubject: Facebook Page unpublish warning\nYour page violated community guidelines. Provide admin credentials to dispute.", "Is this email spam or valid?", ["Spam", "Legitimate"], 0),
        ("From: alerts@discord-nitro-free-claim-gift.tk\nSubject: Discord: Someone gifted you 1 Year of Discord Nitro free!\nClick here to claim before the gift expires in 2 hours.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: security@dropbox-shared-file-download-viewer.cc\nSubject: Confidential financial statement shared with you on Dropbox\nSign in with your email provider to decrypt the file.", "Identify email category:", ["Spam", "Legitimate"], 0),
        ("From: admin@adobe-creative-cloud-license-lapse.top\nSubject: Adobe Acrobat: Your license has expired\nRenew your license for $1.99 by providing your credit card details.", "Is this spam or legitimate?", ["Spam", "Legitimate"], 0),
        ("From: alerts@ebay-account-suspension-appeals.com\nSubject: eBay suspension alert: Counterfeit merchandise reported\nSubmit your bank account statement and password to verify legitimacy.", "Classify this email:", ["Spam", "Legitimate"], 0),
    ]

    # Advance Fee, 419, Lottery & Inheritance Scams (25)
    inheritance_scams = [
        ("From: barrister.musa@lawchambers-nigeria.org\nSubject: CONFIDENTIAL INHERITANCE PROPOSAL: $18.5 MILLION USD\nI am Barrister Musa, personal attorney to a deceased client with your exact surname. With your consent, we can transfer the funds into your bank account. You will receive 40%.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: mrs.fatima.k@un-humanitarian-charity.com\nSubject: DONATION FOR ORPHANS: $4.2 MILLION DOLLARS\nI am dying of esophageal cancer. I want you to distribute my late husband's oil fortune to charitable causes. Reply with your WhatsApp number.", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 0),
        ("From: claim-desk@uk-national-lottery-winner-2026.com\nSubject: OFFICIAL WINNER NOTIFICATION: YOU WON £850,000 POUNDS\nYour email was selected by an automated computer ballot. Send your passport and bank details to claim your prize.", "Categorize this email:", ["Spam", "Legitimate"], 0),
        ("From: general.smith@us-army-peacekeeping-syria.com\nSubject: URGENT ASSISTANCE NEEDED: DIPLOMATIC BOX OF GOLD BARS\nDuring patrol we found two trunk boxes containing $12 million. I need a trusted foreigner to receive them through diplomatic courier.", "Is this spam or ham?", ["Spam", "Legitimate"], 0),
        ("From: dr.john.ebuka@centralbank-transfers-gov.ng\nSubject: FINAL REQUISITION: YOUR OVERDUE CONTRACT PAYMENT OF $10.5M\nThe Presidency has approved the immediate wire transfer. Only a minor $350 processing fee is required via Western Union.", "Classify this message:", ["Spam", "Legitimate"], 0),
        ("From: notification@mega-millions-online-jackpot.net\nSubject: Congratulations! You won second tier prize of $1,000,000\nTo claim your winnings, provide your social security number and postal address.", "What is this communication?", ["Spam", "Legitimate"], 0),
        ("From: prince.al-waleed-fund@royal-charity-grants.org\nSubject: Royal grant donation: $2,500,000 USD approved for you\nSelected randomly from international internet users. Reply immediately with your full banking coordinates.", "Is this email legitimate?", ["Spam", "Legitimate"], 0),
        ("From: auditor.chen@hangseng-bank-claims.hk\nSubject: UNCLAIMED DORMANT ACCOUNT: $22,800,000 USD\nAn abandoned account exists in our branch with no heirs. If you partner with me, we will share the proceeds 50/50.", "Identify email category:", ["Spam", "Legitimate"], 0),
        ("From: customs.inspection@heathrow-airport-diplomatic.co.uk\nSubject: Diplomatic luggage detained with cash funds\nDiplomat Koffi was intercepted with two trunks containing cash. Pay the clearance stamp fee of $500 to release delivery.", "Is this spam?", ["Spam", "Legitimate"], 0),
        ("From: bill-gates-foundation@charity-giveaways-2026.xyz\nSubject: Bill & Melinda Gates Foundation COVID recovery grant\nYou have been awarded $500,000. Send processing fee via Apple gift cards.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: reverend.father.paul@vatican-charity-aid.org\nSubject: Divine blessings and financial empowerment for you\nThe Vatican charitable foundation has selected your email for a $150,000 grant.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: claim-dept@toyota-anniversary-car-sweepstakes.com\nSubject: You won a brand new Toyota Prado Land Cruiser 2026!\nCall our lottery agent immediately to arrange delivery to your home.", "Is this spam or ham?", ["Spam", "Legitimate"], 0),
        ("From: international-monetary-fund@imf-compensation-payout.org\nSubject: IMF Compensation Payment File #9921\nAll victims of internet fraud are being compensated with $2,000,000 ATM card. Pay shipping fee of $150.", "Is this email legitimate?", ["Spam", "Legitimate"], 0),
        ("From: western-union-director@moneytransfer-reimburse.com\nSubject: Western Union: $5,000 daily payout waiting for you\nWe have programmed daily MTCN transfers to your name. Send $85 activation charge.", "What is this communication?", ["Spam", "Legitimate"], 0),
        ("From: petroleum-board@saudi-aramco-subcontract.com\nSubject: Crude oil contract over-invoicing transfer\nWe have an excess balance of $35 million in our contract account. Assist us to move it out of the Middle East.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: unicef-director@united-nations-aid-fund.org\nSubject: Emergency poverty alleviation fund for 2026\nYou were selected to receive $750,000 aid. Send your phone number and copy of ID card.", "Identify email type:", ["Spam", "Legitimate"], 0),
        ("From: swiss-private-banking@credit-suisse-dormant.ch\nSubject: Private matter regarding deceased foreign investor\nAccount balance: 14.7 Million CHF. Looking for a partner with your nationality to file as next of kin.", "Is this spam?", ["Spam", "Legitimate"], 0),
        ("From: fbi.director.wray@investigation-dept-gov.us.com\nSubject: OFFICIAL FBI INVESTIGATION AND PAYMENT ORDER\nThe FBI has recovered your scammed funds. You must pay $200 for certificate of non-involvement.", "What kind of email is this?", ["Spam", "Legitimate"], 0),
        ("From: powerball-lottery@us-lottery-commission-claims.site\nSubject: Official Winner Announcement: Powerball drawing 4410\nYour ticket won $2,000,000. Provide bank details for wire transfer.", "Is this email legitimate?", ["Spam", "Legitimate"], 0),
        ("From: euro-millions-prize@eu-jackpot-winner.eu.com\nSubject: EuroMillions sweepstakes winner: €1,200,000 prize\nPlease contact Mr. Carlos Mendes to process your claim voucher.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: world-bank-aid@wb-compensation-treasury.org\nSubject: World Bank poverty alleviation cash grant\n$1,500,000 approved for credit into your bank account. Reply with your full details.", "Is this spam or valid?", ["Spam", "Legitimate"], 0),
        ("From: express-cargo@diplomatic-delivery-courier.com\nSubject: Consignment parcel with $8.5M cash at airport\nCustoms officer demanding clearance duty of $450 before home delivery.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: attorney-at-law@barrister-chamber-uk.net\nSubject: Estate of late engineer who died in car accident\nNo will found. $9.3M in Barclays Bank. We can claim it as cousins.", "Identify email category:", ["Spam", "Legitimate"], 0),
        ("From: shell-oil-contractor@offshore-contract-surplus.com\nSubject: Surplus contract allocation of $14,000,000\nNeed foreign account to absorb funds from oil pipeline project.", "Is this spam or ham?", ["Spam", "Legitimate"], 0),
        ("From: global-crypto-airdrop@free-bitcoin-multiplier.info\nSubject: Send 0.1 BTC to receive 1.0 BTC immediately back!\nElon Musk giveaway promotion live on Twitter stream.", "Is this email legitimate?", ["Spam", "Legitimate"], 0),
    ]

    # Fake Delivery, Unpaid Fees & Extortion (25)
    fake_delivery_extortion = [
        ("From: parcel-tracking@dhl-customs-unpaid-fee.info\nSubject: DHL Express: Your package is held at customs warehouse\nTo release parcel #991829, pay outstanding duty tax fee of $3.49 within 12 hours or package will be destroyed. Click here to pay: http://dhl-tax-pay.cc", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: delivery-status@fedex-reschedule-delivery-fee.top\nSubject: FedEx: Failed delivery attempt - Incorrect address\nWe attempted to deliver your parcel but no one was home. Pay $2.80 address correction fee to reschedule.", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 0),
        ("From: alerts@usps-tracking-redelivery-attempt.xyz\nSubject: USPS: Package on hold due to missing house number\nUpdate your street address and pay $1.95 redelivery fee at http://usps-redeliver-post.biz.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: tracking@ups-parcel-on-hold-customs.site\nSubject: UPS: Unpaid customs charges for package 1Z992182\nYour international package cannot be delivered until customs tariff of $4.15 is settled online.", "Identify email type:", ["Spam", "Legitimate"], 0),
        ("From: hacker-syndicate@darkweb-blackmail-group.org\nSubject: I recorded you through your webcam while watching adult sites\nI have infected your operating system with Pegasus spyware. Send $1,500 in Bitcoin to wallet 1A1zP1... or I will email the video to all your Facebook contacts.", "Is this email spam/extortion?", ["Spam", "Legitimate"], 0),
        ("From: legal@dmca-copyright-violation-penalty.top\nSubject: Copyright Infringement Notice: Fine of $5,000 pending\nWe detected torrent downloads on your IP. Pay $250 settlement fine to avoid legal subpoena.", "What kind of email is this?", ["Spam", "Legitimate"], 0),
        ("From: traffic-fines@city-municipal-speed-camera.online\nSubject: Speed Camera Violation #992182 - Unpaid ticket\nSpeed detected: 52 mph in 35 mph zone. Pay fine immediately before driver license suspension.", "Is this spam?", ["Spam", "Legitimate"], 0),
        ("From: geek-squad-billing@geeksquad-renew-invoice.xyz\nSubject: INVOICE: Geek Squad Total Tech Support auto-renewal $499.99\nThank you for renewing your annual protection plan. If you did not authorize this, call 1-888-FAKE-NUM immediately.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: mcafee-antivirus-service@mcafee-renew-billing.site\nSubject: Renewal Invoice: McAfee Total Protection $399.00 debited\nYour subscription has auto-renewed. Call customer billing department within 24 hours to cancel refund.", "Is this email spam or valid?", ["Spam", "Legitimate"], 0),
        ("From: norton-lifelock-renew@norton-support-billing.club\nSubject: Order Receipt: Norton 360 Premium protection $299.99\nPayment processed successfully via checking account. Dial 1-800-FAKE-NUM to reverse charges.", "What is this communication?", ["Spam", "Legitimate"], 0),
        ("From: package@royalmail-delivery-fee-update.cc\nSubject: Royal Mail: Redelivery fee required (£1.45)\nYour parcel reference #GB88192 is held at local sorting office. Pay fee to release.", "Is this spam or ham?", ["Spam", "Legitimate"], 0),
        ("From: parcel@dpd-reschedule-driver-fee.top\nSubject: DPD: We missed you today! Fee of €2.50 required\nDriver could not access building. Enter payment details to arrange next-day delivery.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: track@chronopost-frais-douane.info\nSubject: Chronopost: Frais de douane en attente pour votre colis\nVeuillez régler 2,90 € pour débloquer votre livraison avant ce soir.", "Ce courriel est-il un spam?", ["Spam", "Legitimate"], 0),
        ("From: paket@dhl-nachsendung-gebuehr.site\nSubject: DHL Paket: Adressangaben unvollständig - 1,99 € Gebühr\nIhr Paket konnte nicht zugestellt werden. Bitte aktualisieren Sie Ihre Adresse hier.", "Ist diese E-Mail Spam?", ["Spam", "Legitimate"], 0),
        ("From: envio@correos-tasas-aduanas.online\nSubject: Correos: Pago de tasas arancelarias pendientes (2,64 €)\nTu paquete internacional está retenido en la aduana de Madrid Barajas.", "Es este correo spam?", ["Spam", "Legitimate"], 0),
        ("From: speed-ticket@dept-of-transportation-tickets.org\nSubject: Notice of Delinquent Traffic Violation\nYour car was photographed running a red light. Pay $150 fine or warrant will be issued.", "What is this email?", ["Spam", "Legitimate"], 0),
        ("From: irs-tax-audit@internal-revenue-service-gov.cc\nSubject: FINAL NOTICE: Overdue tax balance of $4,890.00\nFederal agents will be dispatched to your residence if payment is not remitted today.", "Is this communication spam?", ["Spam", "Legitimate"], 0),
        ("From: toll-roads@e-zpass-unpaid-toll-violation.top\nSubject: E-ZPass: Toll invoice past due - $8.50\nUnpaid toll crossing on I-95. Settle online to avoid late penalties and vehicle registration block.", "Identify email category:", ["Spam", "Legitimate"], 0),
        ("From: energy-billing@coned-power-disconnection-alert.xyz\nSubject: URGENT: Electric power disconnection scheduled in 4 hours\nYour utility bill is delinquent. Pay $180 immediately via MoneyPak voucher.", "Is this email legitimate?", ["Spam", "Legitimate"], 0),
        ("From: court-subpoena@state-district-court-notice.biz\nSubject: Subpoena to Appear in Court as Witness\nDownload court_notice_case_99182.zip and review charges before trial date.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: job-offer@work-from-home-amazon-tasks.club\nSubject: Earn $500/day rating products from home on your phone\nNo experience needed! 1 hour per day. Send $50 starter registration kit fee.", "Is this email spam or valid?", ["Spam", "Legitimate"], 0),
        ("From: mystery-shopper@walmart-evaluation-program.top\nSubject: Walmart Mystery Shopper Assignment: $3,000 check enclosed\nDeposit the cashier check, keep $400, and wire the rest to our evaluation agent.", "What kind of email is this?", ["Spam", "Legitimate"], 0),
        ("From: loan-approval@instant-cash-loans-no-credit.site\nSubject: Guaranteed loan approval for $25,000 USD\nNo credit check required. Interest rate 1%. Pay $200 insurance fee to release funds.", "Is this spam?", ["Spam", "Legitimate"], 0),
        ("From: pharmacy@canadian-online-meds-discount.org\nSubject: Buy generic Viagra, Cialis and Xanax without prescription\n80% discount on all pills. Discreet international shipping directly to your door.", "Email classification:", ["Spam", "Legitimate"], 0),
        ("From: weight-loss@miracle-keto-diet-secret.xyz\nSubject: Lose 30 lbs in 2 weeks without exercise or dieting\nCelebrity doctor reveals strange morning ritual that dissolves belly fat overnight.", "What is this communication?", ["Spam", "Legitimate"], 0),
    ]

    # Multi-Language Spam & Phishing (25)
    multilingual_spam = [
        ("From: seguridad@banco-santander-alerta-login.top\nSubject: Banco Santander: Su cuenta ha sido bloqueada temporalmente\nHemos detectado accesos sospechosos desde el extranjero. Ingrese sus claves y tarjeta de coordenadas aquí para desbloquear.", "Clasifica este correo:", ["Spam", "Legitimate"], 0),
        ("From: soporte@bbva-verificar-identidad-portal.site\nSubject: BBVA: Actualización obligatoria del sistema de seguridad\nEstimado cliente, actualice su número de teléfono y contraseña o su cuenta será suspendida en 24 horas.", "¿Es este correo spam o legítimo?", ["Spam", "Legitimate"], 0),
        ("From: hacienda@agenciatributaria-devolucion-impuestos.online\nSubject: Reembolso de impuestos pendiente: 382,45 €\nLa Agencia Tributaria ha determinado una devolución a su favor. Ingrese su número de tarjeta para recibir el ingreso.", "Identifica la categoría de este mensaje:", ["Spam", "Legitimate"], 0),
        ("From: dgt-multas@sede-electronica-dgt-notificaciones.top\nSubject: DGT: Notificación de sanción de tráfico no pagada\nDispone de 24 horas para pagar la multa con descuento del 50%. Descargue el expediente adjunto multa.exe.", "¿Es este correo spam?", ["Spam", "Legitimate"], 0),
        ("From: seguridad@caixabank-alerta-tarjeta.xyz\nSubject: CaixaBank: Su tarjeta de débito ha sido restringida\nPara reactivar sus compras online, ingrese su código PIN de 4 dígitos en nuestro enlace seguro.", "Clasificación del correo:", ["Spam", "Legitimate"], 0),
        ("From: alerte@banque-postale-securite-connexion.top\nSubject: La Banque Postale: Votre accès en ligne est temporairement suspendu\nVeuillez confirmer vos identifiants et votre mot de passe pour lever le blocage.", "Ce courriel est-il un spam ou légitime?", ["Spam", "Legitimate"], 0),
        ("From: service-client@societe-generale-securite-maj.site\nSubject: Société Générale: Nouveau dispositif de sécurité obligatoire\nConnectez-vous à votre espace client pour valider vos coordonnées bancaires.", "Quel est ce message?", ["Spam", "Legitimate"], 0),
        ("From: impot-gouv@remboursement-fiscal-tresor.online\nSubject: Avis de remboursement d'impôt sur le revenu de 420,50 €\nRenseignez vos numéros de carte bancaire pour obtenir le virement sur votre compte.", "Classification de l'email:", ["Spam", "Legitimate"], 0),
        ("From: securite@credit-agricole-alerte-fraude.xyz\nSubject: Crédit Agricole: Tentative de virement frauduleux détectée\nUn virement de 850 € vers l'Espagne est en cours. Annulez-le immédiatement ici.", "Ce courriel est-il un spam?", ["Spam", "Legitimate"], 0),
        ("From: livraison@mondial-relay-colis-bloque.club\nSubject: Mondial Relay: Votre colis n'a pas pu être déposé au point relais\nPayez 1,95 € pour relancer la livraison vers un autre point.", "Est-ce un spam ou légitime?", ["Spam", "Legitimate"], 0),
        ("From: sicherheit@sparkasse-onlinebanking-sperrung.top\nSubject: Sparkasse: Wichtige Sicherheitsmitteilung bezüglich Ihres Kontos\nIhr pushTAN-Verfahren läuft in 48 Stunden ab. Erneuern Sie Ihre Zugangsdaten hier.", "Ist diese E-Mail Spam oder legitim?", ["Spam", "Legitimate"], 0),
        ("From: kundendienst@deutsche-bank-konto-reaktivierung.site\nSubject: Deutsche Bank: Ihr Konto wurde aus Sicherheitsgründen eingeschränkt\nBitte verifizieren Sie Ihre PIN und Telefonnummer, um Einschränkungen aufzuheben.", "Klassifizieren Sie diese E-Mail:", ["Spam", "Legitimate"], 0),
        ("From: finanzamt@bundeszentralamt-steuererstattung.online\nSubject: Rückerstattung Ihrer Lohnsteuer für 2025: 412,80 EUR\nGeben Sie Ihre Kreditkartendaten ein, um den Betrag gutzuschreiben.", "Ist diese Nachricht Spam?", ["Spam", "Legitimate"], 0),
        ("From: sicherheit@postbank-legitimation-update.xyz\nSubject: Postbank: BestSign Sicherheitszertifikat abgelaufen\nKlicken Sie hier zur sofortigen Aktualisierung Ihres Online-Bankings.", "Was für eine E-Mail ist das?", ["Spam", "Legitimate"], 0),
        ("From: alerts@sbi-yono-kyc-pan-update.top\nSubject: SBI YONO Alert: Your account will be suspended today\nDear Customer, your SBI account KYC is expired. Update your PAN card and Aadhaar details immediately on http://sbi-kyc-pan-update.site to avoid account freeze.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: service@hdfc-netbanking-reward-points.site\nSubject: HDFC Bank: Claim your 9,850 credit card reward points worth Rs 9,850\nPoints expire tonight. Enter your credit card number, CVV and OTP to redeem directly as cash to bank.", "Is this email spam or legitimate?", ["Spam", "Legitimate"], 0),
        ("From: lottery@kbc-jio-lottery-winner-25lakh.xyz\nSubject: KBC Jio Lottery winner 25 Lakh Rupees congratulations!\nDear WhatsApp user, you won 25 Lakh in Kaun Banega Crorepati. Contact manager Rana Pratap Singh on WhatsApp to claim.", "What is this communication?", ["Spam", "Legitimate"], 0),
        ("From: tax-refund@incometax-gov-refund-credited.top\nSubject: Income Tax Department: Tax refund of Rs 24,500 approved\nRefund rejected due to incorrect bank IFSC. Click here to enter debit card details and PIN for instant IMPS transfer.", "Is this communication spam?", ["Spam", "Legitimate"], 0),
        ("From: electricity@bses-delhi-power-bill-disconnection.online\nSubject: BSES Electricity: Your power supply will be disconnected at 9:30 PM\nPrevious month bill unpaid. Call electricity officer immediately on 9812981290 to pay via Paytm.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: jobs@work-from-home-typing-salary-daily.info\nSubject: Simple data entry job: Earn Rs 3,000 per day working 2 hours\nPayment guarantee. Pay registration fee of Rs 499 to receive question papers.", "Is this spam or ham?", ["Spam", "Legitimate"], 0),
        ("From: loan@instant-personal-loan-preapproved-5lakh.biz\nSubject: Pre-approved instant loan of Rs 5,00,000 disbursed to your account\nNo documents required. Low interest. Pay Rs 1,500 file processing charges.", "What kind of email is this?", ["Spam", "Legitimate"], 0),
        ("From: crypto@telegram-crypto-pump-signals-vip.site\nSubject: VIP Trading Signals: Turn $100 into $10,000 in 7 days\nJoin our private Russian hacker group on Telegram before spots fill up.", "Is this spam?", ["Spam", "Legitimate"], 0),
        ("From: winner@amazon-spin-the-wheel-iphone16.xyz\nSubject: You won iPhone 16 Pro Max in Amazon Diwali spin contest!\nClaim your prize for just Rs 99 shipping fee. Pay with Google Pay now.", "Classify this email:", ["Spam", "Legitimate"], 0),
        ("From: support@uber-free-rides-voucher-code.club\nSubject: Claim $100 Uber ride voucher for free\nAnswer 3 survey questions and provide your mobile phone number to enter draw.", "Is this message spam or valid?", ["Spam", "Legitimate"], 0),
        ("From: security@apple-id-fr-alerte-perte.top\nSubject: Apple: Votre iPhone perdu a été localisé à Paris\nCliquez sur la carte et saisissez votre mot de passe iCloud pour voir l'emplacement en direct.", "Ce message est-il un spam?", ["Spam", "Legitimate"], 0),
    ]

    for item in credential_phishing + inheritance_scams + fake_delivery_extortion + multilingual_spam:
        examples.append({
            "category": "Spam & Phishing",
            "context": item[0],
            "question": item[1],
            "options": item[2],
            "target": item[3]
        })

    # =========================================================================
    # CATEGORY 3: GENERAL FACTS & REASONING (100 SAMPLES)
    # =========================================================================

    # Science & Nature (25)
    science = [
        ("The process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water is fundamental to life on Earth.", "What biological process is described?", ["Respiration", "Photosynthesis", "Fermentation", "Transpiration"], 1),
        ("Water has a chemical formula composed of two hydrogen atoms and one oxygen atom.", "What is the chemical formula of water?", ["CO2", "H2O", "CH4", "NaCl"], 1),
        ("Light travels in a vacuum at a universal physical constant velocity.", "Approximately what is the speed of light in a vacuum?", ["300,000 km/s", "150,000 km/s", "30,000 km/s", "3,000 km/s"], 0),
        ("The Earth completes one full orbit around the Sun in approximately one solar year.", "How long does Earth take to orbit the Sun?", ["365.25 days", "30 days", "24 hours", "687 days"], 0),
        ("The force of attraction that exists between any two masses in the universe is formulated by Newton's law.", "What force keeps planets in orbit around stars?", ["Electromagnetism", "Strong nuclear force", "Gravity", "Frictional force"], 2),
        ("Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions.", "What is the primary function of mitochondria?", ["Protein synthesis", "Cellular energy production (ATP)", "Cell division", "Photosynthesis"], 1),
        ("The human heart has four distinct chambers that pump oxygen-depleted and oxygen-rich blood throughout the body.", "How many chambers does the human heart have?", ["Two", "Three", "Four", "Six"], 2),
        ("Water freezes into ice under standard atmospheric pressure at sea level.", "At what temperature Celsius does pure water freeze?", ["100°C", "0°C", "-32°C", "32°C"], 1),
        ("DNA is the molecule that carries the genetic instructions used in the development and functioning of all known living organisms.", "What does DNA stand for?", ["Deoxyribonucleic acid", "Diribonucleic acid", "Deoxynitrogenous acid", "Decarbonated acid"], 0),
        ("The closest star to Earth is located at the center of our solar system.", "What is the closest star to our planet?", ["Proxima Centauri", "Sirius", "The Sun", "Betelgeuse"], 2),
        ("The Earth's atmosphere is composed primarily of two gases, with one gas making up about 78% by volume.", "Which gas is most abundant in Earth's atmosphere?", ["Oxygen", "Carbon Dioxide", "Nitrogen", "Argon"], 2),
        ("Sound waves require a physical medium such as air, liquid, or solid matter to propagate through vibrations.", "Can sound travel through a vacuum in outer space?", ["Yes, easily", "No, it requires a medium", "Only radio sound waves can", "Only at the speed of light"], 1),
        ("The periodic table arranges elements by their atomic numbers. Gold has atomic number 79.", "What is the chemical symbol for Gold?", ["Ag", "Au", "Fe", "Pb"], 1),
        ("Table salt is a mineral composed primarily of sodium chloride.", "What is the chemical formula for common table salt?", ["NaCl", "KCl", "NaOH", "HCl"], 0),
        ("The human skeleton in an adult typically consists of a specific number of articulated bones.", "How many bones are in an adult human body?", ["150", "206", "300", "512"], 1),
        ("The red color of human blood is due to an iron-containing protein in red blood cells that transports oxygen.", "What is this oxygen-binding protein called?", ["Insulin", "Hemoglobin", "Keratin", "Collagen"], 1),
        ("Mars is the fourth planet from the Sun and appears reddish due to iron oxide on its surface.", "What is Mars commonly known as?", ["The Blue Planet", "The Gas Giant", "The Red Planet", "The Morning Star"], 2),
        ("A diamond is a solid form of a pure element with its atoms arranged in a crystal structure.", "Which chemical element makes up diamond?", ["Carbon", "Silicon", "Nitrogen", "Sulfur"], 0),
        ("Earthquakes are measured on logarithmic scales based on seismic energy released.", "What instrument is used to detect and record earthquakes?", ["Barometer", "Seismograph", "Anemometer", "Hydrometer"], 1),
        ("The largest mammal currently living on Earth is a marine mammal found in oceans worldwide.", "What is the largest living mammal?", ["African Elephant", "Blue Whale", "Sperm Whale", "Giraffe"], 1),
        ("Electrons carry a fundamental negative electric charge.", "What electric charge do electrons carry?", ["Positive", "Negative", "Neutral", "Variable"], 1),
        ("The ozone layer in the stratosphere absorbs the majority of ultraviolet radiation from the Sun.", "Which atmospheric layer contains the ozone layer?", ["Troposphere", "Stratosphere", "Mesosphere", "Thermosphere"], 1),
        ("Newton's third law of motion describes paired interaction forces between bodies.", "What does Newton's third law state?", ["F = ma", "Every action has an equal and opposite reaction", "Energy cannot be destroyed", "Inertia maintains velocity"], 1),
        ("Penicillin was the first effective modern antibiotic, discovered in 1928.", "Who discovered penicillin?", ["Louis Pasteur", "Alexander Fleming", "Marie Curie", "Robert Koch"], 1),
        ("The pH scale measures how acidic or basic a substance is, with pure water at 7.", "What pH value represents a neutral solution?", ["0", "7", "14", "10"], 1),
    ]

    # Geography & History (25)
    geography_history = [
        ("Australia is a country and continent surrounded by the Indian and Pacific oceans. Its federal capital was chosen as a compromise between Sydney and Melbourne.", "What is the capital city of Australia?", ["Sydney", "Melbourne", "Canberra", "Brisbane"], 2),
        ("Canada is the second largest country in the world by land area, divided into ten provinces and three territories.", "What is the capital city of Canada?", ["Toronto", "Montreal", "Ottawa", "Vancouver"], 2),
        ("The longest river in South America flows across the continent through a massive rainforest basin into the Atlantic Ocean.", "What is the longest river in South America?", ["Amazon River", "Nile River", "Yangtze River", "Mississippi River"], 0),
        ("Mount Everest is the highest mountain peak above sea level on Earth, situated in the Himalayas.", "What is the highest mountain above sea level on Earth?", ["K2", "Mount Everest", "Kangchenjunga", "Mont Blanc"], 1),
        ("Japan is an island nation located in East Asia. Its capital is the most populous metropolitan area in the world.", "What is the capital of Japan?", ["Kyoto", "Osaka", "Tokyo", "Hiroshima"], 2),
        ("The French Revolution was a period of political and societal change in France that began with the storming of the Bastille in a specific year.", "In what year did the French Revolution begin?", ["1776", "1789", "1804", "1815"], 1),
        ("Apollo 11 was the American spaceflight that first landed humans on the Moon in July 1969.", "Who was the first human to walk on the Moon?", ["Buzz Aldrin", "Neil Armstrong", "Yuri Gagarin", "Michael Collins"], 1),
        ("World War II in Europe came to an end in 1945 following the surrender of Nazi Germany.", "In what year did World War II conclude?", ["1918", "1939", "1945", "1953"], 2),
        ("The Roman Empire was one of the largest empires in ancient history, centered around the Mediterranean Sea.", "What was the official language of ancient Rome?", ["Greek", "Latin", "Italian", "Aramaic"], 1),
        ("The Sahara Desert is the largest hot desert in the world, spanning most of northern Africa.", "On which continent is the Sahara Desert located?", ["Asia", "Africa", "South America", "Australia"], 1),
        ("Brazil is the largest country in South America and the only Portuguese-speaking nation in the Americas.", "What is the official language of Brazil?", ["Spanish", "Portuguese", "French", "English"], 1),
        ("The United States declared its independence from Great Britain in 1776 in Philadelphia.", "In what year was the US Declaration of Independence signed?", ["1492", "1776", "1789", "1865"], 1),
        ("The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra.", "In which country is the Taj Mahal located?", ["Pakistan", "India", "Bangladesh", "Nepal"], 1),
        ("The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.", "In which country is the Great Wall located?", ["Japan", "China", "Korea", "Mongolia"], 1),
        ("The Colosseum is an oval amphitheatre in the centre of the city of Rome.", "In which country is the Colosseum located?", ["Greece", "Italy", "Spain", "France"], 1),
        ("The Panama Canal connects two major oceans across the Isthmus of Panama.", "Which two oceans are connected by the Panama Canal?", ["Atlantic and Pacific", "Atlantic and Indian", "Pacific and Arctic", "Indian and Southern"], 0),
        ("Egypt is home to the famous Giza pyramid complex and the Great Sphinx.", "What is the capital of Egypt?", ["Alexandria", "Cairo", "Luxor", "Giza"], 1),
        ("The first President of the United States served from 1789 to 1797.", "Who was the first President of the United States?", ["Thomas Jefferson", "George Washington", "John Adams", "Abraham Lincoln"], 1),
        ("The Berlin Wall divided the city of Berlin from 1961 until its fall during the peaceful revolution.", "In what year did the Berlin Wall fall?", ["1989", "1991", "1975", "1968"], 0),
        ("The Magna Carta was issued in 1215 by King John of England, establishing the principle that everyone is subject to the law.", "In which country was Magna Carta signed?", ["France", "England", "Germany", "Scotland"], 1),
        ("The deepest known oceanic trench on Earth is located in the western Pacific Ocean.", "What is the deepest ocean trench on Earth?", ["Java Trench", "Mariana Trench", "Puerto Rico Trench", "Sunda Trench"], 1),
        ("The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth that began in Florence.", "In which country did the Renaissance originate?", ["France", "England", "Italy", "Netherlands"], 2),
        ("The United Nations was established following the end of World War II to maintain international peace.", "In what year was the United Nations founded?", ["1919", "1945", "1955", "1960"], 1),
        ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars.", "In which city is the Eiffel Tower located?", ["London", "Paris", "Berlin", "Rome"], 1),
        ("The Amazon rainforest is the world's largest tropical rainforest. The majority of the forest is contained within one country.", "Which country contains the largest portion of the Amazon rainforest?", ["Colombia", "Brazil", "Peru", "Venezuela"], 1),
    ]

    # Computer Science & Technology (25)
    cs_tech = [
        ("In computer science, hash tables provide key-value lookups with expected constant time complexity.", "What is the average time complexity of a hash map lookup?", ["O(1)", "O(n)", "O(log n)", "O(n^2)"], 0),
        ("Binary search requires the input array to be ordered to achieve logarithmic search time.", "What precondition must an array satisfy for binary search?", ["Must be sorted", "Must be all positive numbers", "Must contain no duplicates", "Must be a linked list"], 0),
        ("Python is a popular programming language created by Guido van Rossum and released in 1991.", "Who created the Python programming language?", ["Dennis Ritchie", "Guido van Rossum", "Bjarne Stroustrup", "James Gosling"], 1),
        ("Git is a distributed version control system designed to track changes in source code during software development.", "Who initially created the Git version control system?", ["Linus Torvalds", "Bill Gates", "Steve Wozniak", "Ken Thompson"], 0),
        ("HTTP status codes communicate the result of an HTTP request. The 404 code indicates that the server cannot find the requested resource.", "What does HTTP status code 404 represent?", ["OK", "Unauthorized", "Not Found", "Internal Server Error"], 2),
        ("In relational databases, SQL is the standard domain-specific language used for managing data.", "What does SQL stand for?", ["Standard Query Logic", "Structured Query Language", "Sequential Query List", "Simple Query Layout"], 1),
        ("A byte is a unit of digital information that most commonly consists of eight bits.", "How many bits are in one standard byte?", ["4", "8", "16", "32"], 1),
        ("The Linux kernel is a free and open-source, monolithic, modular, multitasking Unix-like operating system kernel.", "Who created the Linux kernel?", ["Richard Stallman", "Linus Torvalds", "Steve Jobs", "Andrew Tanenbaum"], 1),
        ("JSON is an open standard file format and data interchange format that uses human-readable text to store and transmit data objects.", "What does JSON stand for?", ["Java Standard Object Notation", "JavaScript Object Notation", "Joined Secure Object Network", "JavaScript Open Notation"], 1),
        ("In object-oriented programming, creating a new class from an existing class to inherit properties is called inheritance.", "What OOP concept allows a child class to inherit behavior from a parent class?", ["Polymorphism", "Encapsulation", "Inheritance", "Abstraction"], 2),
        ("TCP is a core protocol of the Internet protocol suite providing reliable, ordered delivery of a stream of octets.", "What does TCP stand for?", ["Transmission Control Protocol", "Transport Channel Path", "Terminal Connection Protocol", "Transfer Communication Packet"], 0),
        ("RAM is a form of computer memory that can be read and changed in any order, used to store working data and machine code.", "What does RAM stand for?", ["Read Access Module", "Random Access Memory", "Rapid Action Memory", "Run Automated Memory"], 1),
        ("A compiler translates code written in a high-level programming language into machine code or bytecode.", "What is the primary role of a compiler?", ["Run code line-by-line", "Translate source code to executable machine code", "Format text documents", "Manage database connections"], 1),
        ("DNS translates human-readable domain names like example.com into numerical IP addresses.", "What is the main function of DNS?", ["Encrypt passwords", "Resolve domain names to IP addresses", "Compress video streams", "Block malware viruses"], 1),
        ("The C programming language was developed at Bell Labs in the early 1970s.", "Who developed the C programming language?", ["Dennis Ritchie", "Alan Turing", "Tim Berners-Lee", "Grace Hopper"], 0),
        ("An API allows two different software programs to communicate and exchange data with each other.", "What does API stand for?", ["Automated Program Interface", "Application Programming Interface", "App Protocol Integration", "Applied Program Instruction"], 1),
        ("In web development, CSS is used to describe the presentation and layout of a document written in HTML.", "What does CSS stand for?", ["Creative Style Syntax", "Cascading Style Sheets", "Computer Style Structure", "Cascading Syntax Schema"], 1),
        ("A stack is an abstract data type that follows the Last-In, First-Out (LIFO) principle.", "What order of operations does a Stack follow?", ["FIFO (First In First Out)", "LIFO (Last In First Out)", "Random access", "Sorted order"], 1),
        ("A queue is an abstract data structure that follows the First-In, First-Out (FIFO) principle.", "What order does a Queue follow?", ["FIFO (First In First Out)", "LIFO (Last In First Out)", "Priority only", "Double-ended LIFO"], 0),
        ("HTTPS adds security to standard HTTP by using encryption protocols.", "What protocol provides encryption in HTTPS?", ["FTP", "TLS / SSL", "SMTP", "UDP"], 1),
        ("The World Wide Web was invented in 1989 at CERN by a British computer scientist.", "Who invented the World Wide Web?", ["Tim Berners-Lee", "Vint Cerf", "Marc Andreessen", "Bob Kahn"], 0),
        ("IPv4 addresses are 32-bit numbers, while IPv6 addresses use 128 bits.", "How many bits are used in an IPv6 address?", ["32 bits", "64 bits", "128 bits", "256 bits"], 2),
        ("A GPU is specialized electronic circuit designed to manipulate and alter memory to accelerate the creation of images and parallel computing.", "What does GPU stand for?", ["General Processing Unit", "Graphics Processing Unit", "Global Programmable Unit", "Gigabyte Processor Unit"], 1),
        ("Machine learning models often use backpropagation to calculate gradients of a loss function.", "What algorithm is used to compute gradients in neural network training?", ["Dijkstra's algorithm", "Backpropagation", "Quicksort", "A* search"], 1),
        ("In version control with Git, saving changes to the local repository history creates a commit.", "What command records staged changes to the Git repository?", ["git commit", "git push", "git fetch", "git pull"], 0),
    ]

    # Multilingual QA & Logic / Commonsense (25)
    multilingual_logic = [
        ("París es la capital y la ciudad más poblada de Francia, atravesada por el río Sena.", "¿Cuál es la capital de Francia?", ["Marsella", "Lyon", "París", "Burdeos"], 2),
        ("España es un país soberano ubicado en el suroeste de Europa, cuya capital y ciudad más grande es Madrid.", "¿Cuál es la capital de España?", ["Barcelona", "Madrid", "Valencia", "Sevilla"], 1),
        ("El agua hierve a nivel del mar a una temperatura fija en la escala Celsius.", "¿A qué temperatura hierve el agua al nivel del mar?", ["50°C", "80°C", "100°C", "120°C"], 2),
        ("El Sol es la estrella central del sistema solar, alrededor de la cual orbitan la Tierra y los demás planetas.", "¿Qué es el Sol?", ["Un planeta", "Una estrella", "Un asteroide", "Una galaxia"], 1),
        ("La Luna es el único satélite natural de la Tierra y gira alrededor de ella.", "¿Qué es la Luna?", ["Un planeta enano", "El satélite natural de la Tierra", "Una estrella fugaz", "Un cometa"], 1),
        ("Berlin ist die Bundeshauptstadt und bevölkerungsreichste Stadt Deutschlands.", "Was ist die Hauptstadt von Deutschland?", ["München", "Hamburg", "Berlin", "Köln"], 2),
        ("Die Erde ist der dritte Planet von der Sonne aus gesehen im Sonnensystem.", "Der wievielte Planet von der Sonne aus ist die Erde?", ["Der erste", "Der zweite", "Der dritte", "Der fünfte"], 2),
        ("Ein Jahr hat normalerweise 365 Tage, in Schaltjahren sind es 366 Tage.", "Wie viele Tage hat ein normales Kalenderjahr?", ["360 Tage", "365 Tage", "366 Tage", "370 Tage"], 1),
        ("Wasser hat im flüssigen Zustand bei 4 Grad Celsius seine höchste Dichte.", "In welchem Aggregatzustand befindet sich Wasser bei 20°C?", ["Gasförmig", "Flüssig", "Fest", "Plasma"], 1),
        ("Rome est la capitale de l'Italie depuis 1871, située au centre de la péninsule italienne.", "Quelle est la capitale de l'Italie?", ["Milan", "Rome", "Naples", "Florence"], 1),
        ("Le français est la langue officielle de la République française.", "Quelle est la langue officielle de la France?", ["L'espagnol", "Le français", "L'allemand", "L'italien"], 1),
        ("Un triangle est une figure géométrique fermée composée de trois côtés et de trois angles.", "Combien de côtés a un triangle?", ["Deux", "Trois", "Quatre", "Cinq"], 1),
        ("L'été est la saison la plus chaude de l'année dans l'hémisphère nord.", "Quelle saison succède au printemps?", ["L'hiver", "L'automne", "L'été", "La mousson"], 2),
        ("भारत की राजधानी नई दिल्ली है, जो देश के उत्तरी भाग में स्थित है।", "भारत की राजधानी कौन सा शहर है?", ["मुंबई", "कोलकाता", "नई दिल्ली", "चेन्नई"], 2),
        ("सूर्य पूर्व दिशा से उदय होता है और पश्चिम दिशा में अस्त होता है।", "सूर्य किस दिशा से निकलता है?", ["उत्तर", "दक्षिण", "पूर्व", "पश्चिम"], 2),
        ("सप्ताह में कुल सात दिन होते हैं, जो सोमवार से शुरू होते हैं।", "एक सप्ताह में कितने दिन होते हैं?", ["पांच", "छह", "सात", "आठ"], 2),
        ("If all roses are flowers and all flowers are plants, what can be logically deduced about roses?", "What logical conclusion follows?", ["All roses are plants", "No roses are plants", "All plants are roses", "Some flowers are roses"], 0),
        ("If an object has a mass of 10 kilograms, and mass remains constant regardless of location, what is its mass on the Moon?", "What is the mass on the Moon?", ["10 kg", "1.6 kg", "60 kg", "0 kg"], 0),
        ("A car travels at a constant speed of 60 miles per hour for 3 hours without stopping.", "How far has the car traveled?", ["120 miles", "180 miles", "240 miles", "90 miles"], 1),
        ("If today is Wednesday, what day of the week will it be in exactly 7 days?", "What day will it be?", ["Tuesday", "Wednesday", "Thursday", "Friday"], 1),
        ("Which of the following is an antonym (opposite meaning) of the word 'transparent'?", "What is the antonym of 'transparent'?", ["Clear", "Opaque", "Luminous", "Glassy"], 1),
        ("Which of the following is a prime number?", "Which number is prime?", ["4", "9", "13", "15"], 2),
        ("If a rectangle has a length of 5 meters and a width of 4 meters, what is its surface area?", "What is the area of the rectangle?", ["9 square meters", "18 square meters", "20 square meters", "25 square meters"], 2),
        ("What is the square root of the number 81?", "What is the square root of 81?", ["7", "8", "9", "11"], 2),
        ("If you mix equal parts of red paint and blue paint, what secondary color is produced?", "What color is created?", ["Green", "Purple", "Orange", "Brown"], 1),
    ]

    for item in science + geography_history + cs_tech + multilingual_logic:
        examples.append({
            "category": "General Facts",
            "context": item[0],
            "question": item[1],
            "options": item[2],
            "target": item[3]
        })

    return examples


def run_benchmark(model_dir: str = "models", prefer_fp32: bool = True, escalation_threshold: float = 0.10):
    print("=" * 75)
    print("  ZESYDE COMPREHENSIVE 300-SAMPLE HANDPICKED BENCHMARK")
    print("=" * 75)
    print(f"Model Directory       : {model_dir}")
    print(f"Precision             : {'FP32 (Full Precision)' if prefer_fp32 else 'INT8 (Quantized)'}")
    print(f"Escalation Threshold  : {escalation_threshold}")
    print("Loading LanguageRouter engine...")

    t0_load = time.perf_counter()
    router = LanguageRouter.from_checkpoints(
        multilingual_path=model_dir,
        prefer_fp32=prefer_fp32,
        default_key="multilingual"
    )
    print(f"Model loaded in {(time.perf_counter() - t0_load):.2f}s.\n")

    dataset = get_handpicked_dataset()
    print(f"Loaded {len(dataset)} evaluation samples across categories:")
    categories = sorted(list(set(d["category"] for d in dataset)))
    for cat in categories:
        cnt = sum(1 for d in dataset if d["category"] == cat)
        print(f"  - {cat:20s}: {cnt} samples")
    print("-" * 75)

    results = []
    latencies = []
    correct_by_cat = {cat: 0 for cat in categories}
    total_by_cat = {cat: 0 for cat in categories}
    escalated_by_cat = {cat: 0 for cat in categories}

    # Spam/Ham specific metrics (Categories 1 & 2)
    # Target 0: Spam, Target 1: Legitimate
    spam_ham_tp = 0  # correctly predicted spam
    spam_ham_fp = 0  # falsely predicted spam (was legitimate)
    spam_ham_tn = 0  # correctly predicted legitimate
    spam_ham_fn = 0  # falsely predicted legitimate (was spam)

    for idx, sample in enumerate(dataset, 1):
        context = sample["context"]
        question = sample["question"]
        options = sample["options"]
        gold_idx = sample["target"]
        gold_str = options[gold_idx]
        cat = sample["category"]

        res = router.decide(
            context=context,
            question=question,
            options=options,
            question_type="choice",
            escalation_threshold=escalation_threshold
        )

        pred_idx = options.index(res.selected_option) if res.selected_option in options else -1
        is_correct = (pred_idx == gold_idx)
        latencies.append(res.latency_ms)

        total_by_cat[cat] += 1
        if is_correct:
            correct_by_cat[cat] += 1
        if res.should_escalate:
            escalated_by_cat[cat] += 1

        # Track Spam vs Ham confusion matrix
        if cat in ["Legitimate Email", "Spam & Phishing"]:
            # Standardize: Spam is class 0, Legitimate is class 1
            if gold_str == "Spam":
                if res.selected_option == "Spam":
                    spam_ham_tp += 1
                else:
                    spam_ham_fn += 1
            elif gold_str == "Legitimate":
                if res.selected_option == "Legitimate":
                    spam_ham_tn += 1
                else:
                    spam_ham_fp += 1

        results.append({
            "idx": idx,
            "category": cat,
            "is_correct": is_correct,
            "gold": gold_str,
            "pred": res.selected_option,
            "confidence": res.confidence,
            "escalate_prob": res.escalate_probability,
            "should_escalate": res.should_escalate,
            "latency_ms": res.latency_ms
        })

    # Overall Metrics
    total_samples = len(dataset)
    total_correct = sum(correct_by_cat.values())
    overall_acc = (total_correct / total_samples) * 100.0

    print("\n" + "=" * 75)
    print("                    BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"Overall Accuracy       : {total_correct}/{total_samples} ({overall_acc:.2f}%)")
    print(f"Latency per Decision   : Mean {np.mean(latencies):.2f}ms | p50 {np.median(latencies):.2f}ms | p95 {np.percentile(latencies, 95):.2f}ms")
    print("-" * 75)
    print(f"{'Category':25s} | {'Samples':8s} | {'Accuracy':10s} | {'Escalated':10s}")
    print("-" * 75)
    for cat in categories:
        acc = (correct_by_cat[cat] / total_by_cat[cat]) * 100.0
        esc = (escalated_by_cat[cat] / total_by_cat[cat]) * 100.0
        print(f"{cat:25s} | {total_by_cat[cat]:8d} | {acc:9.1f}% | {escalated_by_cat[cat]:3d} ({esc:.1f}%)")
    print("-" * 75)

    # Spam/Ham Diagnostics
    total_emails = total_by_cat.get("Legitimate Email", 0) + total_by_cat.get("Spam & Phishing", 0)
    if total_emails > 0:
        precision_spam = spam_ham_tp / max(1, (spam_ham_tp + spam_ham_fp))
        recall_spam = spam_ham_tp / max(1, (spam_ham_tp + spam_ham_fn))
        f1_spam = 2 * (precision_spam * recall_spam) / max(1e-8, (precision_spam + recall_spam))

        precision_ham = spam_ham_tn / max(1, (spam_ham_tn + spam_ham_fn))
        recall_ham = spam_ham_tn / max(1, (spam_ham_tn + spam_ham_fp))
        f1_ham = 2 * (precision_ham * recall_ham) / max(1e-8, (precision_ham + recall_ham))

        print("\n--- Spam vs. Legitimate (Ham) Diagnostic Matrix ---")
        print(f"True Positives (Spam caught)         : {spam_ham_tp} / {spam_ham_tp + spam_ham_fn}")
        print(f"False Positives (Legit flagged spam) : {spam_ham_fp} / {spam_ham_tn + spam_ham_fp}")
        print(f"Spam Detection Precision / Recall    : {precision_spam * 100:.1f}% / {recall_spam * 100:.1f}% (F1: {f1_spam:.3f})")
        print(f"Legitimate Precision / Recall        : {precision_ham * 100:.1f}% / {recall_ham * 100:.1f}% (F1: {f1_ham:.3f})")

    # Escalation Analysis
    mean_esc_correct = np.mean([r["escalate_prob"] for r in results if r["is_correct"]])
    mean_esc_wrong = np.mean([r["escalate_prob"] for r in results if not r["is_correct"]]) if any(not r["is_correct"] for r in results) else 0.0
    print("\n--- Escalation Head Uncertainty Analysis ---")
    print(f"Mean Escalation Risk on CORRECT predictions  : {mean_esc_correct:.4f}")
    if any(not r["is_correct"] for r in results):
        print(f"Mean Escalation Risk on WRONG predictions    : {mean_esc_wrong:.4f}")
        print(f"Risk Separation Ratio (Wrong / Correct)      : {(mean_esc_wrong / max(1e-8, mean_esc_correct)):.2f}x")

    # Sample Misclassifications
    errors = [r for r in results if not r["is_correct"]]
    if errors:
        print(f"\n--- Misclassifications ({len(errors)} samples) ---")
        for err in errors[:5]:
            orig = dataset[err["idx"] - 1]
            first_line = orig["context"].split("\n")[0][:60]
            print(f"Sample #{err['idx']} [{err['category']}]: '{first_line}...'")
            print(f"  Gold: {err['gold']} | Predicted: {err['pred']} (Conf: {err['confidence'] * 100:.1f}%) | Escalate Risk: {err['escalate_prob']:.3f}")
    else:
        print("\n100% Accuracy across all 300 test samples!")

    # Save Results to JSON
    out_path = PROJECT_ROOT / "data" / "handpicked_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": round(overall_acc, 2),
            "total_samples": total_samples,
            "categories": {cat: {"accuracy": round((correct_by_cat[cat] / total_by_cat[cat]) * 100, 2), "total": total_by_cat[cat]} for cat in categories},
            "latency_ms": {"mean": round(float(np.mean(latencies)), 2), "p95": round(float(np.percentile(latencies, 95)), 2)},
            "details": results
        }, f, indent=2)
    print(f"\n[Saved] Detailed evaluation log written to: {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 300-sample handpicked benchmark on Zesyde.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing the model artifacts.")
    parser.add_argument("--int8", action="store_true", default=False, help="Run on INT8 quantized ONNX model instead of FP32.")
    parser.add_argument("--escalation-threshold", type=float, default=0.10, help="Escalation risk threshold (default: 0.10).")
    args = parser.parse_args()

    run_benchmark(
        model_dir=args.model_dir,
        prefer_fp32=not args.int8,
        escalation_threshold=args.escalation_threshold
    )
