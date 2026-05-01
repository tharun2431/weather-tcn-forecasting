// ONNX Model Edge Inference Engine
console.log("Loading DeepWeather Edge Inference Engine...");

let session = null;
let scalerData = null;

// The exact 14 features expected by the PyTorch model
const FEATURES = [
  "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", 
  "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", 
  "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"
];

async function initEdgeModel() {
  try {
    const res = await fetch('scaler.json');
    scalerData = await res.json();
    console.log('Scaler parameters loaded:', scalerData.feature_names);
    
    // Configure ONNX WebAssembly paths explicitly to prevent Emscripten load errors
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    
    // Create ONNX inference session
    session = await ort.InferenceSession.create('lstm_model.onnx');
    console.log('Deep Learning ONNX Session created successfully!');
    return true;
  } catch (e) {
    console.error("Failed to load edge AI model:", e);
    return false;
  }
}

// Ensure array exactly matches python calculation
function buildFeatures(hourly, startIndex, count) {
  const sequence = [];
  
  for (let i = startIndex; i < startIndex + count; i++) {
    const T = hourly.temperature_2m[i];
    const p = hourly.surface_pressure[i];
    const rh = hourly.relative_humidity_2m[i];
    const Tdew = hourly.dew_point_2m[i];
    
    let wv = hourly.wind_speed_10m[i] / 3.6; // km/h to m/s
    if (wv < 0) wv = 0;
    
    let max_wv = hourly.wind_gusts_10m[i] / 3.6;
    if (isNaN(max_wv)) max_wv = wv;
    if (max_wv < 0) max_wv = 0;
    
    const wd = hourly.wind_direction_10m[i];
    
    // Thermodynamic equations matching Python `app.py`
    const T_K = T + 273.15;
    const Tpot = T_K * Math.pow((1000 / p), 0.286);
    const VPmax = 6.112 * Math.exp((17.67 * T) / (T + 243.5));
    const VPact = VPmax * (rh / 100);
    const VPdef = VPmax - VPact;
    const sh = 622 * VPact / (p - 0.378 * VPact);
    const H2OC = (VPact / p) * 1000;
    const Tv = T_K * (1 + (sh / 1000) * 0.61);
    const rho = (p * 100 / (287.05 * Tv)) * 1000;
    
    const rawFeatures = [p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max_wv, wd];
    
    // Apply StandardScaler
    const scaled = rawFeatures.map((val, idx) => (val - scalerData.mean[idx]) / scalerData.scale[idx]);
    sequence.push(scaled);
  }
  return sequence;
}

async function runEdgeInference(hourlyData, currentHourIndex, hoursAhead = 12) {
  if (!session || !scalerData) {
    console.warn("Model not ready. Did you call initEdgeModel()?");
    return null;
  }
  
  const predictions = [];
  
  // Predict using the rolling 168-hour window from the API data 
  // (which includes the API's own future forecasts for other variables)
  for (let step = 0; step < hoursAhead; step++) {
    // We want to predict for `currentHourIndex + 1 + step`
    // So the sequence must end at `currentHourIndex + step`
    const startIndex = currentHourIndex - 167 + step;
    if (startIndex < 0 || startIndex + 168 > hourlyData.temperature_2m.length) {
      console.warn("Not enough data to populate sequence for step", step);
      break;
    }
    
    // Build the 168x14 sequence using the actual/forecasted data from Open-Meteo
    let seq = buildFeatures(hourlyData, startIndex, 168);
    const flatSeq = Float32Array.from(seq.flat());
    const tensor = new ort.Tensor('float32', flatSeq, [1, 168, 14]);
    
    const results = await session.run({ "input": tensor });
    const predScaled = results.temperature.data[0];
    const targetIdx = 1; // "T (degC)"
    
    // Inverse transform
    let realTemp = (predScaled * scalerData.scale[targetIdx]) + scalerData.mean[targetIdx];
    
    // Edge AI standalone prediction (no blending with API)
    // const apiTemp = hourlyData.temperature_2m[startIndex + 167];
    // realTemp = (realTemp * 0.35) + ((apiTemp || realTemp) * 0.65);
    
    predictions.push(realTemp);
  }
  
  return predictions;
}
