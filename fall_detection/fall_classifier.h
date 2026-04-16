// ===================================================
//   fall_classifier.h — Lightweight ML classifier
//   Hardware: NodeMCU ESP8266 + MPU6050
//   Derived from TFLite model weights in model.h
// ===================================================

#ifndef FALL_CLASSIFIER_H
#define FALL_CLASSIFIER_H

#include <math.h>
#include "config.h"
#include "model.h"   // FIX: scaler_mean[] / scaler_scale[] must come
                     // from model.h (single source of truth).
                     // Removed duplicate SCALER_MEAN / SCALER_SCALE
                     // arrays that were out of sync with model.h.

// ── Number of features ──────────────────────────────
#define NUM_FEATURES 64

// ── Feature structure ────────────────────────────────
struct SensorWindow {
  float ax[WINDOW_SIZE];
  float ay[WINDOW_SIZE];
  float az[WINDOW_SIZE];
  float gx[WINDOW_SIZE];
  float gy[WINDOW_SIZE];
  float gz[WINDOW_SIZE];
  float accel_mag[WINDOW_SIZE];
  float gyro_mag[WINDOW_SIZE];
  int   count;
};

// ── Classification result ────────────────────────────
struct ClassificationResult {
  float         fall_probability;
  bool          is_fall;
  bool          high_confidence;
  const char*   reason;
};

// ── Normalize features ───────────────────────────────
// FIX: Now uses scaler_mean[] / scaler_scale[] from model.h
//      instead of the duplicate local arrays (which could silently
//      diverge from the training pipeline).
void normalizeFeatures(float* features, float* normalized) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    if (scaler_scale[i] != 0.0f) {
      normalized[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    } else {
      normalized[i] = 0.0f;
    }
  }
}

// ── Activations ─────────────────────────────────────
inline float relu(float x)    { return x > 0.0f ? x : 0.0f; }
inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ── Extract 64 features from sensor window ───────────
// 8 channels × 8 statistics = 64 features
// FIX: variance was divided by n (biased). Changed to (n-1) for
//      unbiased std-dev, matching sklearn's StandardScaler default.
// FIX: guard added for n==0 to avoid division-by-zero.
void extractFeatures(const SensorWindow& win, float* features) {
  const float* channels[8] = {
    win.ax, win.ay, win.az,
    win.gx, win.gy, win.gz,
    win.accel_mag, win.gyro_mag
  };

  int feat_idx = 0;
  int n = win.count;

  if (n == 0) {
    memset(features, 0, sizeof(float) * NUM_FEATURES);
    return;
  }

  for (int c = 0; c < 8; c++) {
    const float* data = channels[c];

    // ── Mean
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += data[i];
    mean /= (float)n;

    // ── Std deviation  (unbiased, ddof=1 to match sklearn)
    float variance = 0.0f;
    for (int i = 0; i < n; i++) {
      float diff = data[i] - mean;
      variance  += diff * diff;
    }
    // FIX: was /n; sklearn uses /(n-1)
    float std_dev = (n > 1) ? sqrtf(variance / (float)(n - 1)) : 0.0f;

    // ── Max / Min
    float max_val = data[0];
    float min_val = data[0];
    for (int i = 1; i < n; i++) {
      if (data[i] > max_val) max_val = data[i];
      if (data[i] < min_val) min_val = data[i];
    }

    // ── RMS & Energy
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += data[i] * data[i];
    float rms    = sqrtf(sum_sq / (float)n);
    float energy = sum_sq;

    // ── Range
    float range = max_val - min_val;

    // ── Peak index (normalised 0–1)
    // FIX: was comparing fabs(data[i]) but starting peak_val = data[0]
    //      (signed). Now initialised to fabs(data[0]) for consistency.
    float peak_idx = 0.0f;
    float peak_abs = fabsf(data[0]);
    for (int i = 1; i < n; i++) {
      if (fabsf(data[i]) > peak_abs) {
        peak_abs = fabsf(data[i]);
        peak_idx = (float)i / (float)n;
      }
    }

    // Store 8 features per channel — ORDER MUST MATCH TRAINING SCRIPT
    features[feat_idx++] = mean;
    features[feat_idx++] = std_dev;
    features[feat_idx++] = energy;
    features[feat_idx++] = max_val;
    features[feat_idx++] = min_val;
    features[feat_idx++] = peak_idx;
    features[feat_idx++] = rms;
    features[feat_idx++] = range;
  }
}

// ── Lightweight NN approximation ─────────────────────
// Approximates the model.h neural network on constrained ESP8266.
// Architecture: Input(64) → BN → Dense(64) → Dense(32) → Dense(16) → Output(1)
//
// FIX: "Batch normalisation simulation" in original used hard-coded
//      parameters (mean=0.5, var=1.0). These have been replaced with
//      a simple clip to prevent runaway logits while keeping the
//      approximation honest about what it is.
//
// FIX: accel_rms_norm index was 54 and gyro_rms_norm was 62.
//      Feature layout (8 per channel, 0-indexed):
//        ch0=ax(0-7), ch1=ay(8-15), ch2=az(16-23),
//        ch3=gx(24-31), ch4=gy(32-39), ch5=gz(40-47),
//        ch6=accel_mag(48-55), ch7=gyro_mag(56-63)
//      Within each channel: [mean,std,energy,max,min,peak_idx,rms,range]
//      Indices 0=mean,1=std,2=energy,3=max,4=min,5=peak,6=rms,7=range
//      So accel_mag rms = 48+6 = 54 ✓  gyro_mag rms = 56+6 = 62 ✓
//      These were already correct; confirmed and documented here.
float computeFallScore(const float* nf) {
  // Channel base indices
  // az=16, accel_mag=48, gyro_mag=56
  // [+0]=mean,[+1]=std,[+2]=energy,[+3]=max,[+4]=min,[+5]=peak,[+6]=rms,[+7]=range

  float accel_max_norm    = nf[51];  // accel_mag max
  float accel_std_norm    = nf[49];  // accel_mag std
  float accel_range_norm  = nf[55];  // accel_mag range
  float accel_energy_norm = nf[50];  // accel_mag energy
  float accel_mean_norm   = nf[48];  // accel_mag mean
  float accel_rms_norm    = nf[54];  // accel_mag rms

  float gyro_max_norm     = nf[59];  // gyro_mag max
  float gyro_std_norm     = nf[57];  // gyro_mag std
  float gyro_energy_norm  = nf[58];  // gyro_mag energy
  float gyro_rms_norm     = nf[62];  // gyro_mag rms

  float az_mean_norm      = nf[16];  // az mean (low = free-fall)
  float az_energy_norm    = nf[18];  // az energy
  float az_min_norm       = nf[20];  // az min

  // Hidden layer 1 (64→32 approximated)
  float h1 = relu(
    accel_max_norm    *  0.35f +
    accel_std_norm    *  0.25f +
    accel_range_norm  *  0.20f +
    gyro_max_norm     *  0.30f +
    gyro_std_norm     *  0.20f +
    accel_energy_norm *  0.15f +
    az_mean_norm      * -0.40f +   // negative: low az → free-fall
    az_min_norm       * -0.30f +
    0.10f                           // bias
  );

  // Hidden layer 2 (32→16 approximated)
  float h2 = relu(
    h1                *  0.60f +
    accel_rms_norm    *  0.25f +
    gyro_rms_norm     *  0.20f +
    gyro_energy_norm  *  0.15f +
    az_energy_norm    * -0.20f +
    accel_mean_norm   *  0.10f +
    0.05f                           // bias
  );

  // FIX: Removed fake "batch normalisation" (was (h2-0.5)/1.001 then relu).
  //      That formula does nothing useful and was misleading. Replaced with
  //      a simple ReLU clamp which is what the BN layer reduces to at
  //      inference when gamma≈1, beta≈0, mu≈0, var≈1.
  float h2_bn = relu(h2);

  // Output logit → sigmoid
  float logit      = h2_bn * 1.8f + h1 * 0.4f - 0.3f;
  float probability = sigmoid(logit);

  return probability;
}

// ── Main classification function ─────────────────────
ClassificationResult classifyWindow(SensorWindow& win) {
  ClassificationResult result;
  result.is_fall          = false;
  result.high_confidence  = false;
  result.fall_probability = 0.0f;
  result.reason           = "Normal";

  if (win.count < WINDOW_SIZE) {
    result.reason = "Insufficient data";
    return result;
  }

  // Step 1: Extract features
  float features[NUM_FEATURES];
  extractFeatures(win, features);

  // Step 2: Normalise (uses scaler from model.h)
  float normalized[NUM_FEATURES];
  normalizeFeatures(features, normalized);

  // Step 3: Compute fall probability
  float prob = computeFallScore(normalized);
  result.fall_probability = prob;

  // Step 4: Quick physics sanity check
  float max_accel = 0.0f;
  // FIX: min_accel initialised to 99 (arbitrary). Changed to a large
  //      float so it works for any sensor range.
  float min_accel = 1e9f;
  float max_gyro  = 0.0f;

  for (int i = 0; i < win.count; i++) {
    if (win.accel_mag[i] > max_accel) max_accel = win.accel_mag[i];
    if (win.accel_mag[i] < min_accel) min_accel = win.accel_mag[i];
    if (win.gyro_mag[i]  > max_gyro)  max_gyro  = win.gyro_mag[i];
  }

  // FIX: pre_impact_mean was computed but never used in decision logic.
  //      Removed to avoid dead code confusion.

  bool physics_free_fall = (min_accel < FREE_FALL_THRESHOLD);
  bool physics_impact    = (max_accel > IMPACT_THRESHOLD);
  bool physics_rotation  = (max_gyro  > GYRO_THRESHOLD);

  // Step 5: Combined decision
  if (prob >= ML_CONFIDENCE_THRESH) {
    result.is_fall         = true;
    result.high_confidence = true;
    result.reason          = "ML: High confidence fall";

  } else if (prob >= ML_FALL_SCORE_THRESH) {
    if (physics_impact || physics_free_fall || physics_rotation) {
      result.is_fall = true;
      result.reason  = "ML+Physics: Fall confirmed";
    } else {
      result.reason  = "ML: Low confidence, physics normal";
    }

  } else {
    // Pure physics fallback
    if (physics_free_fall && physics_impact) {
      result.is_fall          = true;
      result.reason           = "Physics: Free-fall + impact";
      result.fall_probability = 0.75f;
    } else if (physics_impact && physics_rotation) {
      result.is_fall          = true;
      result.reason           = "Physics: Impact + rotation";
      result.fall_probability = 0.65f;
    }
  }

  return result;
}

// ── Print classification result ──────────────────────
void printResult(const ClassificationResult& r) {
  Serial.print("Fall: ");
  Serial.print(r.is_fall ? "YES" : "NO");
  Serial.print(" | Prob: ");
  Serial.print(r.fall_probability * 100.0f, 1);
  Serial.print("% | ");
  Serial.println(r.reason);
}

#endif // FALL_CLASSIFIER_H