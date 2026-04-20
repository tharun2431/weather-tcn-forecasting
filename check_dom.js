const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
  
  await page.goto('http://localhost:8042/', {waitUntil: 'networkidle0', timeout: 5000}).catch(()=>null);
  
  const cityNameNode = await page.evaluate(() => document.getElementById('cityName') ? true : false);
  console.log("Does cityName exist in DOM?:", cityNameNode);
  
  const aiStatusText = await page.evaluate(() => document.getElementById('aiStatusText') ? true : false);
  console.log("Does aiStatusText exist in DOM?:", aiStatusText);
  
  await browser.close();
})();
