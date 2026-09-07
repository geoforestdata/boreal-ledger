(function () {
  const data = window.BOREAL_LEDGER_DATA;

  const qs = (selector) => document.querySelector(selector);

  function formatNumber(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
    return Number(value).toLocaleString("en-CA", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits
    });
  }

  function initMap() {
    const map = L.map("map", {
      scrollWheelZoom: false,
      zoomControl: true
    }).setView(data.aoi.center, 11);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    fetch("data/aoi.geojson")
      .then((response) => response.json())
      .then((geojson) => {
        const layer = L.geoJSON(geojson, {
          style: {
            color: "#153f2f",
            weight: 3,
            fillColor: "#6f8f54",
            fillOpacity: 0.18
          },
          onEachFeature: (feature, lyr) => {
            lyr.bindPopup(`
              <strong>${feature.properties.name}</strong><br>
              Area: ${formatNumber(feature.properties.area_ha_gee, 1)} ha<br>
              Source: ${feature.properties.source}
            `);
          }
        }).addTo(map);
        map.fitBounds(layer.getBounds(), { padding: [26, 26] });
        window.setTimeout(() => {
          map.invalidateSize();
          map.fitBounds(layer.getBounds(), { padding: [26, 26] });
        }, 120);
      });
  }

  function renderCarbonCards() {
    const container = qs("#carbonCards");
    container.innerHTML = data.carbon.map((item) => `
      <article class="carbon-card" style="border-top: 5px solid ${item.color}">
        <h3>${item.source}</h3>
        <p class="note">${item.scope}</p>
        <div class="carbon-value" style="color:${item.color}">${formatNumber(item.carbon, 1)}</div>
        <div class="unit">Mg C/ha</div>
        <p class="note">${item.note}</p>
      </article>
    `).join("");
  }

  function renderCarbonChart() {
    const agb = data.carbon.filter((item) => item.biomass !== null);
    const max = Math.max(...agb.map((item) => item.carbon));
    qs("#carbonChart").innerHTML = agb.map((item) => `
      <div class="bar-row">
        <strong>${item.source}</strong>
        <div class="bar-track" aria-hidden="true">
          <div class="bar-fill" style="width:${(item.carbon / max) * 100}%; background:${item.color}"></div>
        </div>
        <span>${formatNumber(item.carbon, 1)}</span>
      </div>
    `).join("");
  }

  function renderAttribution() {
    const max = data.attribution.nonFireResidualHa;
    qs("#disturbanceBars").innerHTML = `
      <div class="bar-row">
        <strong>Non-fire residual</strong>
        <div class="bar-track"><div class="bar-fill" style="width:100%"></div></div>
        <span>${formatNumber(data.attribution.nonFireResidualHa, 1)} ha</span>
      </div>
      <div class="bar-row">
        <strong>Mapped fire</strong>
        <div class="bar-track"><div class="bar-fill fire-fill" style="width:${max ? 0 : 0}%"></div></div>
        <span>${formatNumber(data.attribution.fireHa, 1)} ha</span>
      </div>
    `;
  }

  function renderClassChart() {
    const max = Math.max(...data.biomassClasses.flatMap((row) => [row.mrnf, row.esa, row.scanfi]));
    qs("#classChart").innerHTML = data.biomassClasses.map((row) => `
      <div class="class-row">
        <strong>${row.cls}</strong>
        <div class="class-bars">
          <div class="mini-bar mrnf" style="width:${(row.mrnf / max) * 100}%"></div>
          <div class="mini-bar esa" style="width:${(row.esa / max) * 100}%"></div>
          <div class="mini-bar scanfi" style="width:${(row.scanfi / max) * 100}%"></div>
        </div>
      </div>
    `).join("");
  }

  function renderKeyStats() {
    qs("#datedLoss").textContent = `${formatNumber(data.standStructure.datedHansenCanopyLossPct, 1)}%`;
    qs("#meanLossAge").textContent = `${formatNumber(data.standStructure.meanYearsSinceLoss, 1)} yr`;
    qs("#meanHeight").textContent = `${formatNumber(data.standStructure.meanCanopyHeightM, 1)} m`;
    qs("#recoveryClasses").textContent = `${data.recovery.retainedClasses}/${data.recovery.totalClasses}`;
    qs("#mrnfFootprint").textContent = `${formatNumber(data.commonFootprint.areaHa, 1)} ha`;
    qs("#rmseComparison").textContent = `${formatNumber(data.commonFootprint.scanfiRmse, 1)} vs ${formatNumber(data.commonFootprint.esaRmse, 1)}`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    initMap();
    renderKeyStats();
    renderCarbonCards();
    renderCarbonChart();
    renderAttribution();
    renderClassChart();
  });
})();
