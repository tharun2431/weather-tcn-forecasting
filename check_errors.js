const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
  
  try {
    await page.goto('http://localhost:8042/', {waitUntil: 'networkidle0', timeout: 10000});
  } catch (e) {
    console.log("Timeout waiting for network idle", e.message);
  }
  
  const html = await page.content();
  const isLoadingVisible = await page.evaluate(() => document.getElementById('loadingScreen').style.display !== 'none');
  console.log("Is Loading Screen Visible:", isLoadingVisible);
  
  await browser.close();
})();
