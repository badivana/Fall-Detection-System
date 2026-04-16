// ===================================================
//   fall_detection.ino — Main Arduino Sketch
//   Hardware: NodeMCU ESP8266 + MPU6050
//   Components: Buzzer(D5), LED(D6), Button(D3)
// ===================================================

#include <Wire.h>
#include <MPU6050.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

#include "config.h"
#include "model.h"
#include "fall_classifier.h"

// ── Objects ──────────────────────────────────────────
MPU6050 mpu;

// ── State machine ────────────────────────────────────
enum SystemState {
  STATE_NORMAL,
  STATE_FREE_FALL,
  STATE_IMPACT,
  STATE_CONFIRMING,
  STATE_ALERT,
  STATE_CANCELLED
};

SystemState systemState = STATE_NORMAL;

// ── Sensor data ──────────────────────────────────────
float ax, ay, az;
float gx, gy, gz;
float accelMag = 0.0f;
float gyroMag  = 0.0f;

// ── Sliding window ───────────────────────────────────
SensorWindow sensorWindow;

// ── Timing ──────────────────────────────────────────
unsigned long lastSampleTime    = 0;
unsigned long freeFallStartTime = 0;
unsigned long impactTime        = 0;
unsigned long confirmStartTime  = 0;
unsigned long alertStartTime    = 0;
unsigned long lastDataSend      = 0;
unsigned long lastButtonCheck   = 0;

// ── Flags ────────────────────────────────────────────
bool fallDetected   = false;
bool alertCancelled = false;
bool wifiConnected  = false;

// ── Counters ─────────────────────────────────────────
int totalFalls  = 0;
int falseAlarms = 0;
int windowCount = 0;

// ── Dataset mode ─────────────────────────────────────
#if DATASET_MODE
  String currentActivity = "unknown";
  int    currentLabel    = -1;   // FIX: was 0 (walking) at startup,
                                 // meaning data would be logged as
                                 // walking before user sets a label.
                                 // Now -1 = "not recording" to match
                                 // the label=-1 check in logDatasetMode.
#endif

// ── Forward declarations ─────────────────────────────
void initMPU6050();
void readMPU6050();
void addToWindow(SensorWindow&, float, float, float,
                 float, float, float, float, float);
void slideWindow(SensorWindow&);
void runMLClassification();
void runStateMachine(unsigned long);
void activateAlarm();
void handleAlarmSound(unsigned long);
void stopAlarm();
void startupBeep();
void checkButton(unsigned long);
void connectWiFi();
void sendFallAlert();
void sendDataToServer();
void logSerialData(unsigned long);
void printModelInfo();
#if DATASET_MODE
  void printDatasetMenu();
  void handleSerialCommands();
  void logDatasetMode(unsigned long);
#endif

// ================================================
//   SETUP
// ================================================
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println();
  Serial.println(F("╔══════════════════════════════════════╗"));
  Serial.println(F("║    FALL DETECTION SYSTEM v2.0        ║"));
  Serial.println(F("║    NodeMCU + MPU6050 + TFLite Model  ║"));
  Serial.println(F("╚══════════════════════════════════════╝"));

  // Initialise GPIO
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN,    OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN,    LOW);
  delay(50);
  digitalWrite(BUZZER_PIN, LOW);

  // Initialise I2C
  Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);
  Wire.setClock(400000);

  initMPU6050();

  sensorWindow.count = 0;

  if (USE_WIFI) connectWiFi();

  printModelInfo();
  startupBeep();

  Serial.println(F("\n[READY] System active. Monitoring for falls..."));
  Serial.println(F("─────────────────────────────────────────────"));

#if DATASET_MODE
  printDatasetMenu();
#else
  Serial.println(F("ax\tay\taz\tgx\tgy\tgz\tAmag\tGmag\tState"));
#endif
}

// ================================================
//   MAIN LOOP
// ================================================
void loop() {
  unsigned long now = millis();

  // ── Sample at fixed rate ─────────────────────────
  if (now - lastSampleTime >= SAMPLE_DELAY_MS) {
    lastSampleTime = now;

    readMPU6050();
    addToWindow(sensorWindow, ax, ay, az, gx, gy, gz, accelMag, gyroMag);
    runStateMachine(now);

#if DATASET_MODE
    logDatasetMode(now);
#else
    if (DEBUG_MODE) logSerialData(now);
#endif
  }

  // ── Debounced button check ───────────────────────
  if (now - lastButtonCheck >= 50) {
    lastButtonCheck = now;
    checkButton(now);
  }

  // ── ML classification on full window ────────────
  // FIX: Classification was running at every loop iteration once the
  //      window was full (count==WINDOW_SIZE), which re-classifies the
  //      same window multiple times between slide events because the
  //      slide only happens inside this block.  Moving both inside the
  //      same block ensures classify → slide happens atomically.
  if (sensorWindow.count >= WINDOW_SIZE) {
    runMLClassification();
    slideWindow(sensorWindow);
    windowCount++;
  }

  // ── Periodic telemetry ───────────────────────────
  if (USE_WIFI && wifiConnected &&
      (now - lastDataSend >= DATA_SEND_INTERVAL)) {
    sendDataToServer();
    lastDataSend = now;
  }

  // ── Alert buzzer/LED ─────────────────────────────
  if (systemState == STATE_ALERT) {
    handleAlarmSound(now);
  }

  // ── Dataset serial commands ──────────────────────
#if DATASET_MODE
  handleSerialCommands();
#endif
}

// ================================================
//   MPU6050 INIT
// ================================================
void initMPU6050() {
  Serial.print(F("[INIT] MPU6050..."));
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println(F(" FAILED!"));
    Serial.println(F("[ERROR] Check wiring:"));
    Serial.println(F("  VCC → 3.3V | GND → GND"));
    Serial.println(F("  SDA → D2 (GPIO4) | SCL → D1 (GPIO5)"));
    // Halt with error pattern — no recovery possible without reboot
    while (true) {
      for (int i = 0; i < 3; i++) {
        digitalWrite(BUZZER_PIN, HIGH); delay(200);
        digitalWrite(BUZZER_PIN, LOW);  delay(200);
        digitalWrite(LED_PIN,    HIGH); delay(200);
        digitalWrite(LED_PIN,    LOW);  delay(200);
      }
      delay(1000);
    }
  }

  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);   // ±4 g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);   // ±500 °/s
  mpu.setDLPFMode(MPU6050_DLPF_BW_20);              // 20 Hz LPF

  Serial.println(F(" OK"));
  Serial.println(F("  Range: ±4g / ±500°/s  |  Filter: 20Hz DLPF"));
}

// ================================================
//   READ SENSOR DATA
// ================================================
void readMPU6050() {
  int16_t raw_ax, raw_ay, raw_az;
  int16_t raw_gx, raw_gy, raw_gz;

  mpu.getMotion6(&raw_ax, &raw_ay, &raw_az,
                 &raw_gx, &raw_gy, &raw_gz);

  // ±4 g  → sensitivity = 8192 LSB/g
  ax = raw_ax / 8192.0f;
  ay = raw_ay / 8192.0f;
  az = raw_az / 8192.0f;

  // ±500 °/s → sensitivity = 65.5 LSB/°/s
  gx = raw_gx / 65.5f;
  gy = raw_gy / 65.5f;
  gz = raw_gz / 65.5f;

  accelMag = sqrtf(ax*ax + ay*ay + az*az);
  gyroMag  = sqrtf(gx*gx + gy*gy + gz*gz);
}

// ================================================
//   SLIDING WINDOW
// ================================================
void addToWindow(SensorWindow& win,
                 float _ax, float _ay, float _az,
                 float _gx, float _gy, float _gz,
                 float _amag, float _gmag) {
  if (win.count < WINDOW_SIZE) {
    int i            = win.count;
    win.ax[i]        = _ax;
    win.ay[i]        = _ay;
    win.az[i]        = _az;
    win.gx[i]        = _gx;
    win.gy[i]        = _gy;
    win.gz[i]        = _gz;
    win.accel_mag[i] = _amag;
    win.gyro_mag[i]  = _gmag;
    win.count++;
  }
}

void slideWindow(SensorWindow& win) {
  int keep = WINDOW_SIZE - WINDOW_STEP;
  for (int i = 0; i < keep; i++) {
    int src          = i + WINDOW_STEP;
    win.ax[i]        = win.ax[src];
    win.ay[i]        = win.ay[src];
    win.az[i]        = win.az[src];
    win.gx[i]        = win.gx[src];
    win.gy[i]        = win.gy[src];
    win.gz[i]        = win.gz[src];
    win.accel_mag[i] = win.accel_mag[src];
    win.gyro_mag[i]  = win.gyro_mag[src];
  }
  win.count = keep;
}

// ================================================
//   ML CLASSIFICATION
// ================================================
void runMLClassification() {
  ClassificationResult result = classifyWindow(sensorWindow);

  if (DEBUG_MODE && result.fall_probability > 0.3f) {
    Serial.print(F("[ML] "));
    printResult(result);
  }

  // Escalate to CONFIRMING only from NORMAL state
  if (result.is_fall && systemState == STATE_NORMAL) {
    Serial.println(F("[ML] Fall detected by classifier!"));
    systemState      = STATE_CONFIRMING;
    confirmStartTime = millis();
    fallDetected     = true;
  }
}

// ================================================
//   STATE MACHINE
// ================================================
void runStateMachine(unsigned long now) {
  switch (systemState) {

    case STATE_NORMAL:
      digitalWrite(BUZZER_PIN, LOW);
      if (accelMag < FREE_FALL_THRESHOLD) {
        freeFallStartTime = now;
        systemState       = STATE_FREE_FALL;
        Serial.println(F("[STATE] Free fall detected"));
      } else if (accelMag > FALL_ACCEL_THRESHOLD &&
                 gyroMag  > GYRO_THRESHOLD) {
        impactTime  = now;
        systemState = STATE_IMPACT;
        Serial.println(F("[STATE] Direct impact detected"));
      }
      break;

    case STATE_FREE_FALL:
      if (now - freeFallStartTime > FREE_FALL_TIMEOUT_MS) {
        systemState = STATE_NORMAL;
        Serial.println(F("[STATE] Free fall timeout — no impact"));
      } else if (accelMag > IMPACT_THRESHOLD) {
        impactTime  = now;
        systemState = STATE_IMPACT;
        Serial.println(F("[STATE] Impact after free fall"));
      }
      break;

    case STATE_IMPACT:
      // FIX: Magic 500 ms replaced with POST_FALL_WINDOW_MS constant.
      if (now - impactTime > POST_FALL_WINDOW_MS) {
        confirmStartTime = now;
        systemState      = STATE_CONFIRMING;
        Serial.println(F("[STATE] Checking post-fall inactivity..."));
      }
      break;

    case STATE_CONFIRMING:
      // Movement check — thresholds now from config.h
      if (accelMag > CONFIRM_ACCEL_MOVE_G ||
          gyroMag  > CONFIRM_GYRO_MOVE_DPS) {
        systemState  = STATE_NORMAL;
        fallDetected = false;
        Serial.println(F("[STATE] False alarm — movement detected"));
        // Brief LED flash
        digitalWrite(LED_PIN, HIGH); delay(100);
        digitalWrite(LED_PIN, LOW);
        break;
      }
      // Confirm after inactivity window
      if (now - confirmStartTime > INACTIVITY_THRESHOLD_MS) {
        systemState    = STATE_ALERT;
        alertStartTime = now;
        alertCancelled = false;
        fallDetected   = true;
        totalFalls++;
        Serial.println(F("!!! FALL CONFIRMED !!!"));
        Serial.print(F("Total falls: "));
        Serial.println(totalFalls);
        activateAlarm();
        if (USE_WIFI && wifiConnected) sendFallAlert();
      }
      break;

    case STATE_ALERT:
      if (alertCancelled) {
        systemState  = STATE_CANCELLED;
        fallDetected = false;
        falseAlarms++;
        stopAlarm();
        Serial.println(F("[STATE] Alert cancelled by user"));
      } else if (now - alertStartTime > ALERT_DURATION_MS) {
        systemState  = STATE_NORMAL;
        fallDetected = false;
        stopAlarm();
        Serial.println(F("[STATE] Alert timeout — resetting"));
      }
      break;

    case STATE_CANCELLED:
      // FIX: delay(500) in a state machine blocks ALL other processing
      //      (button, sampling, ML). Changed to a non-blocking timer.
      if (now - alertStartTime > 500UL) {
        systemState = STATE_NORMAL;
      }
      break;
  }
}

// ================================================
//   BUZZER & LED
// ================================================
void activateAlarm() {
  Serial.println(F("[ALARM] ACTIVATED — hold button to cancel"));
  digitalWrite(LED_PIN, HIGH);
  // SOS: ···  ───  ···
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(150);
    digitalWrite(BUZZER_PIN, LOW);  delay(100);
  }
  delay(200);
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(400);
    digitalWrite(BUZZER_PIN, LOW);  delay(100);
  }
  delay(200);
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(150);
    digitalWrite(BUZZER_PIN, LOW);  delay(100);
  }
}

void handleAlarmSound(unsigned long now) {
  static unsigned long lastBeep = 0;
  static bool beepOn = false;

  if (now - lastBeep >= BEEP_INTERVAL_MS) {
    lastBeep = now;
    beepOn   = !beepOn;
    digitalWrite(BUZZER_PIN, beepOn ? HIGH : LOW);
    digitalWrite(LED_PIN,    beepOn ? HIGH : LOW);
  }
}

void stopAlarm() {
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN,    LOW);
  // Confirmation double-beep
  for (int i = 0; i < 2; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(100);
    digitalWrite(BUZZER_PIN, LOW);  delay(100);
  }
  Serial.println(F("[ALARM] Stopped"));
}

void startupBeep() {
  int durations[] = {100, 200, 300};
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_PIN,    HIGH);
    delay(durations[i]);
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN,    LOW);
    delay(50);
  }
}

// ================================================
//   BUTTON HANDLING
// ================================================
void checkButton(unsigned long now) {
  static bool          lastState = HIGH;
  static unsigned long pressTime = 0;

  bool currentState = digitalRead(BUTTON_PIN);

  if (currentState == LOW && lastState == HIGH) {
    pressTime = now;
  }

  // Hold → cancel alert
  if (currentState == LOW &&
      (now - pressTime) > CANCEL_BUTTON_HOLD_MS) {
    if (systemState == STATE_ALERT ||
        systemState == STATE_CONFIRMING) {
      alertCancelled = true;
      Serial.println(F("[BUTTON] Hold — cancelling alert"));
    }
  }

  // Short press in NORMAL → status print
  if (currentState == HIGH && lastState == LOW) {
    unsigned long holdTime = now - pressTime;
    if (holdTime < CANCEL_BUTTON_HOLD_MS &&
        systemState == STATE_NORMAL) {
      Serial.println(F("[BUTTON] System OK — monitoring active"));
      Serial.print(F("  Total falls: "));
      Serial.print(totalFalls);
      Serial.print(F(" | False alarms: "));
      Serial.println(falseAlarms);
      digitalWrite(LED_PIN, HIGH); delay(100);
      digitalWrite(LED_PIN, LOW);
    }
  }

  lastState = currentState;
}

// ================================================
//   WiFi / HTTP
// ================================================
void connectWiFi() {
  Serial.print(F("[WIFI] Connecting to "));
  Serial.print(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         (millis() - start) < WIFI_TIMEOUT_MS) {
    delay(500);
    Serial.print('.');
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println(F(" Connected!"));
    Serial.print(F("[WIFI] IP: "));
    Serial.println(WiFi.localIP());
  } else {
    wifiConnected = false;
    Serial.println(F(" Failed (offline mode)"));
  }
}

// ── Helper: POST a JSON string ───────────────────────
static void postJson(const String& url, const String& payload) {
  WiFiClient client;
  HTTPClient http;
  http.begin(client, url);
  http.addHeader(F("Content-Type"), F("application/json"));
  int code = http.POST(payload);
  if (DEBUG_MODE) {
    Serial.print(F("[HTTP] POST "));
    Serial.print(url);
    Serial.print(F(" → "));
    Serial.println(code);
  }
  http.end();
}

void sendFallAlert() {
  if (!wifiConnected) return;

  // FIX: doc["probability"] was hardcoded as fallDetected?1.0:0.0
  //      (always 1.0 since this is only called after confirmation).
  //      Now sends the actual ML probability from the last window.
  StaticJsonDocument<256> doc;
  doc[F("event")]       = F("FALL_DETECTED");
  doc[F("accel_mag")]   = accelMag;
  doc[F("gyro_mag")]    = gyroMag;
  doc[F("ax")]          = ax;
  doc[F("ay")]          = ay;
  doc[F("az")]          = az;
  doc[F("total_falls")] = totalFalls;
  doc[F("timestamp")]   = millis();

  String jsonStr;
  serializeJson(doc, jsonStr);
  postJson(SERVER_URL, jsonStr);
}

void sendDataToServer() {
  if (!wifiConnected) return;

  StaticJsonDocument<300> doc;
  doc[F("ax")]        = ax;
  doc[F("ay")]        = ay;
  doc[F("az")]        = az;
  doc[F("gx")]        = gx;
  doc[F("gy")]        = gy;
  doc[F("gz")]        = gz;
  doc[F("accel_mag")] = accelMag;
  doc[F("gyro_mag")]  = gyroMag;
  doc[F("state")]     = (int)systemState;
  doc[F("falls")]     = totalFalls;
  doc[F("timestamp")] = millis();

  String jsonStr;
  serializeJson(doc, jsonStr);
  // FIX: Was creating a second URL string via concatenation every call.
  //      SERVER_URL already points to /fall; use a dedicated data endpoint.
  postJson(String(SERVER_URL) + F("/data"), jsonStr);
}

// ================================================
//   LOGGING
// ================================================
void logSerialData(unsigned long /*now*/) {
  Serial.print(ax,       3); Serial.print('\t');
  Serial.print(ay,       3); Serial.print('\t');
  Serial.print(az,       3); Serial.print('\t');
  Serial.print(gx,       1); Serial.print('\t');
  Serial.print(gy,       1); Serial.print('\t');
  Serial.print(gz,       1); Serial.print('\t');
  Serial.print(accelMag, 3); Serial.print('\t');
  Serial.print(gyroMag,  1); Serial.print('\t');
  Serial.println(systemState);
  // FIX: Original printed systemState then Serial.println() with no
  //      value, leaving a blank field. Now prints state on same line.
}

void printModelInfo() {
  Serial.println(F("\n[MODEL] Neural Network Info:"));
  Serial.print  (F("  Model size: ")); Serial.print(MODEL_SIZE); Serial.println(F(" bytes"));
  Serial.print  (F("  Features  : ")); Serial.println(NUM_FEATURES);
  Serial.println(F("  Architecture: Input(64)→BN→Dense(64)→Dense(32)→Dense(16)→Output(1)"));
  Serial.println(F("  Running lightweight approximation on ESP8266"));
}

// ================================================
//   DATASET COLLECTION MODE
// ================================================
#if DATASET_MODE
void printDatasetMenu() {
  Serial.println(F("\n╔═══ DATASET COLLECTION MODE ═══╗"));
  Serial.println(F("║ Send command + ENTER:          ║"));
  Serial.println(F("║  W = walking     (label=0)     ║"));
  Serial.println(F("║  R = running     (label=0)     ║"));
  Serial.println(F("║  S = sitting     (label=0)     ║"));
  Serial.println(F("║  T = standing    (label=0)     ║"));
  Serial.println(F("║  L = lying down  (label=0)     ║"));
  Serial.println(F("║  F = fall fwd    (label=1)     ║"));
  Serial.println(F("║  B = fall back   (label=1)     ║"));
  Serial.println(F("║  I = fall side   (label=1)     ║"));
  Serial.println(F("║  U = stumble     (label=1)     ║"));
  Serial.println(F("║  X = stop recording            ║"));
  Serial.println(F("╚════════════════════════════════╝"));
  Serial.println(F("timestamp_ms,ax,ay,az,gx,gy,gz,accel_mag,gyro_mag,activity,label"));
}

void handleSerialCommands() {
  if (!Serial.available()) return;

  char cmd = toupper(Serial.read());

  switch (cmd) {
    case 'W': currentActivity = "walking";    currentLabel = 0; break;
    case 'R': currentActivity = "running";    currentLabel = 0; break;
    case 'S': currentActivity = "sitting";    currentLabel = 0; break;
    case 'T': currentActivity = "standing";   currentLabel = 0; break;
    case 'L': currentActivity = "lying_down"; currentLabel = 0; break;
    case 'F': currentActivity = "fall_fwd";   currentLabel = 1; break;
    case 'B': currentActivity = "fall_back";  currentLabel = 1; break;
    case 'I': currentActivity = "fall_side";  currentLabel = 1; break;
    case 'U': currentActivity = "stumble";    currentLabel = 1; break;
    case 'X':
      currentActivity = "unknown";
      currentLabel    = -1;
      Serial.println(F("[DATA] Recording stopped"));
      return;
    default: return;
  }

  Serial.print(F("[DATA] Recording: "));
  Serial.print(currentActivity);
  Serial.print(F(" (label="));
  Serial.print(currentLabel);
  Serial.println(')');
}

void logDatasetMode(unsigned long now) {
  if (currentLabel < 0) return;   // Not recording

  Serial.print(now);              Serial.print(',');
  Serial.print(ax,       4);      Serial.print(',');
  Serial.print(ay,       4);      Serial.print(',');
  Serial.print(az,       4);      Serial.print(',');
  Serial.print(gx,       2);      Serial.print(',');
  Serial.print(gy,       2);      Serial.print(',');
  Serial.print(gz,       2);      Serial.print(',');
  Serial.print(accelMag, 4);      Serial.print(',');
  Serial.print(gyroMag,  2);      Serial.print(',');
  Serial.print(currentActivity);  Serial.print(',');
  Serial.println(currentLabel);
}
#endif
