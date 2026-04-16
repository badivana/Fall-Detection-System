// ===================================================
//   config.h — System-wide configuration
//   Hardware: NodeMCU ESP8266 + MPU6050
// ===================================================

#ifndef CONFIG_H
#define CONFIG_H

// ── PIN DEFINITIONS ──────────────────────────────────
#define BUZZER_PIN        13    // D7
#define BUTTON_PIN         0    // D3 (INPUT_PULLUP, active LOW)
#define MPU_SDA_PIN        4    // D2
#define MPU_SCL_PIN        5    // D1
#define LED_PIN           12    // D6 (active HIGH)

// ── WIFI SETTINGS ────────────────────────────────────
#define WIFI_SSID         "Danna"
#define WIFI_PASSWORD     "12345679"
#define WIFI_TIMEOUT_MS   10000

// ── SERVER ───────────────────────────────────────────
#define SERVER_URL        "http://192.168.1.100:5000/fall"
#define USE_WIFI          false   // Set true to enable WiFi

// ── MPU6050 SAMPLING ─────────────────────────────────
#define SAMPLE_RATE_HZ    50
// FIX: SAMPLE_DELAY_MS must equal exactly 1000/SAMPLE_RATE_HZ.
//      Enforced here as a compile-time expression so it stays
//      in sync if SAMPLE_RATE_HZ is changed.
#define SAMPLE_DELAY_MS   (1000 / SAMPLE_RATE_HZ)   // = 20 ms

#define WINDOW_SIZE       25     // samples per classification window
#define WINDOW_STEP       13     // slide step (~50% overlap)

// Sanity check: step must be < window
static_assert(WINDOW_STEP < WINDOW_SIZE,
              "WINDOW_STEP must be less than WINDOW_SIZE");

// ── PHYSICS THRESHOLDS ───────────────────────────────
// FIX: Original comment on FREE_FALL_THRESHOLD said "lowered to 0.6g"
//      but the value was *raised* from 0.4 g to 0.6 g. Comment corrected.
//      Raising the threshold reduces missed detections (higher sensitivity)
//      because more of the free-fall phase exceeds the threshold.
//      Wait — free-fall detection fires when accelMag < threshold, so a
//      HIGHER threshold means MORE samples qualify as free-fall (more sensitive).
//      Original 0.4 g was conservative; 0.6 g is more sensitive. Correct.
#define FREE_FALL_THRESHOLD      0.6f   // g  — fire when accelMag < this
#define FALL_ACCEL_THRESHOLD     2.0f   // g  — sudden impact in STATE_NORMAL
#define IMPACT_THRESHOLD         2.0f   // g  — impact after free-fall
#define GYRO_THRESHOLD         200.0f   // °/s — significant body rotation

// FIX: CONFIRM_DELAY_MS was defined but the state machine used
//      INACTIVITY_THRESHOLD_MS for the same purpose, creating two
//      constants with identical roles. CONFIRM_DELAY_MS removed;
//      use INACTIVITY_THRESHOLD_MS everywhere.
#define INACTIVITY_THRESHOLD_MS  1500   // ms — post-fall stillness window
#define POST_FALL_WINDOW_MS      2000   // ms — wait after impact before confirming
#define FREE_FALL_TIMEOUT_MS     1000   // ms — max duration of free-fall phase

// ── CONFIRMING STATE MOVEMENT THRESHOLDS ─────────────
#define CONFIRM_ACCEL_MOVE_G    1.3f    // g    — movement → false alarm
#define CONFIRM_GYRO_MOVE_DPS  80.0f    // °/s  — rotation → false alarm

// ── ML THRESHOLDS ────────────────────────────────────
#define ML_FALL_SCORE_THRESH    0.5f    // minimum score to consider a fall
#define ML_CONFIDENCE_THRESH    0.7f    // high-confidence direct classification

// ── ALERT TIMINGS ────────────────────────────────────
#define ALERT_DURATION_MS      30000   // ms — alert auto-timeout
#define CANCEL_BUTTON_HOLD_MS    500   // ms — hold duration to cancel alert
#define BEEP_INTERVAL_MS         500   // ms — alarm beep cadence

// ── SYSTEM ───────────────────────────────────────────
#define SERIAL_BAUD          115200
#define DATA_SEND_INTERVAL     5000   // ms — telemetry send interval

// FIX: DEBUG_MODE and DATASET_MODE were both true simultaneously.
//      In deployment DEBUG_MODE floods the serial port and DATASET_MODE
//      prints CSV headers that corrupt telemetry parsers.
//      Set DATASET_MODE true only during data collection.
//      Set DEBUG_MODE  true only during development/testing.
//      They should never both be true in production firmware.
#define DEBUG_MODE    true    // set false for production
#define DATASET_MODE  false   // set true only for data collection

// Guard: warn if both are enabled together (runtime serial noise)
#if DEBUG_MODE && DATASET_MODE
  #warning "DEBUG_MODE and DATASET_MODE are both enabled. Serial output will be mixed."
#endif

#endif // CONFIG_H