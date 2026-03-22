// Weather Prediction Dashboard - Interactive Charts

document.addEventListener('DOMContentLoaded', function () {
    // Navigation
    setupNavigation();

    // Load data and create charts
    loadDashboard();

    // Setup live prediction
    setupLivePrediction();
});


function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();

            // Remove active class from all
            navLinks.forEach(l => l.classList.remove('active'));
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

            // Add active class
            this.classList.add('active');
            const sectionId = this.getAttribute('data-section');
            document.getElementById(sectionId).classList.add('active');
        });
    });
}


async function loadDashboard() {
    try {
        const [resultsRes, predsRes] = await Promise.all([
            fetch('/api/results'),
            fetch('/api/predictions')
        ]);

        const results = await resultsRes.json();
        const predictions = await predsRes.json();

        createOverviewCharts(results);
        createPredictionChart(predictions);
        createErrorCharts(predictions);
        createComparisonRadar(results);
        createHorizonChart(results);

    } catch (error) {
        console.error('Error loading data:', error);
    }
}


// Chart.js global defaults
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = '#1f2937';
Chart.defaults.font.family = "'Inter', sans-serif";


function createOverviewCharts(results) {
    // R² Score Chart
    const r2Ctx = document.getElementById('r2Chart').getContext('2d');
    new Chart(r2Ctx, {
        type: 'bar',
        data: {
            labels: ['LSTM', 'TCN'],
            datasets: [{
                data: [results.short_term.lstm.R2, results.short_term.tcn.R2],
                backgroundColor: ['rgba(59, 130, 246, 0.7)', 'rgba(16, 185, 129, 0.7)'],
                borderColor: ['#3b82f6', '#10b981'],
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.5,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    min: 98, max: 100, grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { callback: v => v + '%' }
                },
                x: { grid: { display: false } }
            }
        }
    });

    // RMSE Chart
    const rmsCtx = document.getElementById('rmseChart').getContext('2d');
    new Chart(rmsCtx, {
        type: 'bar',
        data: {
            labels: ['LSTM', 'TCN'],
            datasets: [{
                data: [results.short_term.lstm.RMSE, results.short_term.tcn.RMSE],
                backgroundColor: ['rgba(59, 130, 246, 0.7)', 'rgba(16, 185, 129, 0.7)'],
                borderColor: ['#3b82f6', '#10b981'],
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.5,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { callback: v => v + '°C' }
                },
                x: { grid: { display: false } }
            }
        }
    });

    // MAE Chart
    const maeCtx = document.getElementById('maeChart').getContext('2d');
    new Chart(maeCtx, {
        type: 'bar',
        data: {
            labels: ['LSTM', 'TCN'],
            datasets: [{
                data: [results.short_term.lstm.MAE, results.short_term.tcn.MAE],
                backgroundColor: ['rgba(59, 130, 246, 0.7)', 'rgba(16, 185, 129, 0.7)'],
                borderColor: ['#3b82f6', '#10b981'],
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.5,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true, max: 0.7, grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { callback: v => v + '°C' }
                },
                x: { grid: { display: false } }
            }
        }
    });
}


function createPredictionChart(predictions) {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: predictions.hours.slice(0, 150),
            datasets: [
                {
                    label: 'Actual Temperature',
                    data: predictions.actual.slice(0, 150),
                    borderColor: '#f3f4f6',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'LSTM Prediction',
                    data: predictions.lstm.slice(0, 150),
                    borderColor: '#3b82f6',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    borderDash: [5, 3],
                    tension: 0.3,
                },
                {
                    label: 'TCN Prediction',
                    data: predictions.tcn.slice(0, 150),
                    borderColor: '#10b981',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    borderDash: [5, 3],
                    tension: 0.3,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 3,
            plugins: {
                legend: {
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: { size: 12 }
                    }
                }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Temperature (°C)' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                x: {
                    title: { display: true, text: 'Hour' },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 15 }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false,
            }
        }
    });

    // Chart controls
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            const show = this.getAttribute('data-show');
            chart.data.datasets[0].hidden = false;
            chart.data.datasets[1].hidden = show === 'tcn';
            chart.data.datasets[2].hidden = show === 'lstm';
            chart.update();
        });
    });
}


function createErrorCharts(predictions) {
    const actual = predictions.actual.slice(0, 150);

    // LSTM errors
    const lstmErrors = predictions.lstm.slice(0, 150).map((p, i) => p - actual[i]);
    createHistogram('lstmErrorChart', lstmErrors, '#3b82f6', 'LSTM');

    // TCN errors
    const tcnErrors = predictions.tcn.slice(0, 150).map((p, i) => p - actual[i]);
    createHistogram('tcnErrorChart', tcnErrors, '#10b981', 'TCN');
}


function createHistogram(canvasId, errors, color, name) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Create bins
    const bins = 20;
    const min = Math.min(...errors);
    const max = Math.max(...errors);
    const binWidth = (max - min) / bins;
    const binLabels = [];
    const binCounts = new Array(bins).fill(0);

    for (let i = 0; i < bins; i++) {
        binLabels.push((min + i * binWidth).toFixed(1));
    }

    errors.forEach(e => {
        const idx = Math.min(Math.floor((e - min) / binWidth), bins - 1);
        binCounts[idx]++;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: `${name} Error`,
                data: binCounts,
                backgroundColor: color + '66',
                borderColor: color,
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Count' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                x: {
                    title: { display: true, text: 'Error (°C)' },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 }
                }
            }
        }
    });
}


function createComparisonRadar(results) {
    const ctx = document.getElementById('comparisonRadar').getContext('2d');

    // Normalize metrics for radar chart (0-100 scale)
    const maxRMSE = 1.5;
    const maxMAE = 1.0;
    const maxMAPE = 10;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['R² Score', 'Low RMSE', 'Low MAE', 'Low MAPE', 'Training Speed'],
            datasets: [
                {
                    label: 'LSTM',
                    data: [
                        results.short_term.lstm.R2,
                        (1 - results.short_term.lstm.RMSE / maxRMSE) * 100,
                        (1 - results.short_term.lstm.MAE / maxMAE) * 100,
                        (1 - results.short_term.lstm.MAPE / maxMAPE) * 100,
                        65
                    ],
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointBackgroundColor: '#3b82f6',
                },
                {
                    label: 'TCN',
                    data: [
                        results.short_term.tcn.R2,
                        (1 - results.short_term.tcn.RMSE / maxRMSE) * 100,
                        (1 - results.short_term.tcn.MAE / maxMAE) * 100,
                        (1 - results.short_term.tcn.MAPE / maxMAPE) * 100,
                        90
                    ],
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointBackgroundColor: '#10b981',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.5,
            plugins: {
                legend: {
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: { size: 13 }
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    angleLines: { color: 'rgba(255,255,255,0.08)' },
                    pointLabels: { font: { size: 12 } },
                    ticks: { display: false }
                }
            }
        }
    });
}


function createHorizonChart(results) {
    const ctx = document.getElementById('horizonChart').getContext('2d');

    const horizons = ['24h', '48h', '72h'];
    const lstmR2 = horizons.map(h => results.long_term[h].lstm.R2);
    const tcnR2 = horizons.map(h => results.long_term[h].tcn.R2);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['24 hours (1 day)', '48 hours (2 days)', '72 hours (3 days)'],
            datasets: [
                {
                    label: 'LSTM',
                    data: lstmR2,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    borderWidth: 3,
                    pointRadius: 6,
                    pointBackgroundColor: '#3b82f6',
                    tension: 0.3,
                },
                {
                    label: 'TCN',
                    data: tcnR2,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    borderWidth: 3,
                    pointRadius: 6,
                    pointBackgroundColor: '#10b981',
                    tension: 0.3,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: { size: 13 }
                    }
                }
            },
            scales: {
                y: {
                    title: { display: true, text: 'R² Score (%)' },
                    min: 60,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                x: {
                    title: { display: true, text: 'Forecast Horizon' },
                    grid: { display: false }
                }
            }
        }
    });
}


// ============================================
// LIVE PREDICTION
// ============================================
let livePredChart = null;

async function setupLivePrediction() {
    try {
        const res = await fetch('/api/cities');
        const cities = await res.json();
        const grid = document.getElementById('cityGrid');

        for (const [key, city] of Object.entries(cities)) {
            const btn = document.createElement('button');
            btn.className = 'city-btn';
            btn.setAttribute('data-city', key);
            btn.innerHTML = `<span class="city-emoji">${city.emoji}</span>${city.name}`;
            btn.addEventListener('click', () => selectCity(key, cities));
            grid.appendChild(btn);
        }
    } catch (e) {
        console.error('Error loading cities:', e);
    }
}


async function selectCity(cityKey, cities) {
    // Update active button
    document.querySelectorAll('.city-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-city="${cityKey}"]`).classList.add('active');

    // Show loading
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('currentWeather').style.display = 'none';
    document.getElementById('livePredContainer').style.display = 'none';
    document.getElementById('predSummary').style.display = 'none';

    try {
        const res = await fetch(`/api/live_predict?city=${cityKey}`);
        const data = await res.json();

        if (data.error) {
            alert('Error: ' + data.error);
            document.getElementById('loadingIndicator').style.display = 'none';
            return;
        }

        // Hide loading
        document.getElementById('loadingIndicator').style.display = 'none';

        // Show current weather
        document.getElementById('currentWeather').style.display = 'block';
        document.getElementById('cityName').textContent =
            `${data.city.emoji} ${data.city.name} - Current Conditions`;
        document.getElementById('currentTemp').textContent =
            data.current.temperature.toFixed(1) + '\u00B0C';
        document.getElementById('currentHumidity').textContent =
            data.current.humidity.toFixed(0) + '%';
        document.getElementById('currentPressure').textContent =
            data.current.pressure.toFixed(0) + ' hPa';
        document.getElementById('currentWind').textContent =
            data.current.wind_speed.toFixed(1) + ' km/h';

        // Create prediction chart
        createLivePredChart(data);

        // Update prediction summary
        updatePredSummary(data);

    } catch (e) {
        console.error('Error fetching prediction:', e);
        document.getElementById('loadingIndicator').style.display = 'none';
        alert('Failed to fetch weather data. Please try again.');
    }
}


function createLivePredChart(data) {
    document.getElementById('livePredContainer').style.display = 'block';

    const ctx = document.getElementById('livePredChart').getContext('2d');

    // Destroy old chart
    if (livePredChart) {
        livePredChart.destroy();
    }

    // Format time labels
    const pastTimes = data.past.times.map(t => {
        const d = new Date(t);
        return d.getHours() + ':00';
    });
    const futureTimes = data.predictions.times.map(t => {
        const d = new Date(t);
        return d.getHours() + ':00';
    });

    const allLabels = [...pastTimes, ...futureTimes];
    const pastLen = pastTimes.length;

    // Past actual data (fill future with null)
    const pastData = [...data.past.temps, ...new Array(futureTimes.length).fill(null)];

    // Future predictions (fill past with null)
    const lstmData = [...new Array(pastLen).fill(null), ...data.predictions.lstm.values];
    const tcnData = [...new Array(pastLen).fill(null), ...data.predictions.tcn.values];

    // Confidence intervals
    const lstmUpper = [...new Array(pastLen).fill(null), ...data.predictions.lstm.upper];
    const lstmLower = [...new Array(pastLen).fill(null), ...data.predictions.lstm.lower];
    const tcnUpper = [...new Array(pastLen).fill(null), ...data.predictions.tcn.upper];
    const tcnLower = [...new Array(pastLen).fill(null), ...data.predictions.tcn.lower];

    // Actual future if available
    const actualFuture = data.predictions.actual.length > 0
        ? [...new Array(pastLen).fill(null), ...data.predictions.actual]
        : null;

    const datasets = [
        {
            label: 'Past Actual',
            data: pastData,
            borderColor: '#f3f4f6',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: false,
        },
        {
            label: 'LSTM Prediction',
            data: lstmData,
            borderColor: '#3b82f6',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.3,
            fill: false,
        },
        {
            label: 'LSTM Confidence',
            data: lstmUpper,
            borderColor: 'transparent',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
        },
        {
            label: 'LSTM Lower',
            data: lstmLower,
            borderColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.3,
        },
        {
            label: 'TCN Prediction',
            data: tcnData,
            borderColor: '#10b981',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.3,
            fill: false,
        },
        {
            label: 'TCN Confidence',
            data: tcnUpper,
            borderColor: 'transparent',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
        },
        {
            label: 'TCN Lower',
            data: tcnLower,
            borderColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.3,
        }
    ];

    // Add actual future if available
    if (actualFuture) {
        datasets.push({
            label: 'Actual Future',
            data: actualFuture,
            borderColor: '#f59e0b',
            borderWidth: 2,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.3,
            fill: false,
        });
    }

    livePredChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    labels: {
                        filter: item => !['LSTM Confidence', 'LSTM Lower', 'TCN Confidence', 'TCN Lower'].includes(item.text),
                        padding: 20,
                        usePointStyle: true,
                        font: { size: 12 }
                    }
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: pastLen - 1,
                            xMax: pastLen - 1,
                            borderColor: 'rgba(255,255,255,0.3)',
                            borderWidth: 2,
                            borderDash: [6, 6],
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Temperature (\u00B0C)' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                x: {
                    title: { display: true, text: 'Time (hour)' },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 16 }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false,
            }
        }
    });
}


function updatePredSummary(data) {
    document.getElementById('predSummary').style.display = 'grid';

    const tcn = data.predictions.tcn.values;
    const lstm = data.predictions.lstm.values;

    // TCN predictions at key horizons
    document.getElementById('tcnNext1h').textContent = tcn[0] ? tcn[0].toFixed(1) + '\u00B0C' : '--';
    document.getElementById('tcnNext6h').textContent = tcn[5] ? tcn[5].toFixed(1) + '\u00B0C' : '--';
    document.getElementById('tcnNext12h').textContent = tcn[11] ? tcn[11].toFixed(1) + '\u00B0C' : '--';
    document.getElementById('tcnNext24h').textContent = tcn[23] ? tcn[23].toFixed(1) + '\u00B0C' : '--';

    // LSTM predictions at key horizons
    document.getElementById('lstmNext1h').textContent = lstm[0] ? lstm[0].toFixed(1) + '\u00B0C' : '--';
    document.getElementById('lstmNext6h').textContent = lstm[5] ? lstm[5].toFixed(1) + '\u00B0C' : '--';
    document.getElementById('lstmNext12h').textContent = lstm[11] ? lstm[11].toFixed(1) + '\u00B0C' : '--';
    document.getElementById('lstmNext24h').textContent = lstm[23] ? lstm[23].toFixed(1) + '\u00B0C' : '--';
}
