window.BOREAL_LEDGER_DATA = {
  aoi: {
    name: "Abitibi-Temiscamingue managed boreal landscape",
    areaHa: 10009.5,
    forestedAreaHa: 8971.7,
    center: [48.36951136817797, -79.54678860036347],
    bbox: [-79.61448860036347, 48.32451136817797, -79.47908860036347, 48.414511368177976]
  },
  standStructure: {
    datedHansenCanopyLossPct: 10.1,
    meanYearsSinceLoss: 14.6,
    meanCanopyHeightM: 17.7,
    canopyHeightNote: "Canopy height is structural information only; it is not stand age."
  },
  attribution: {
    nonFireResidualHa: 1015.5,
    fireHa: 0,
    nbacPerimetersIntersecting: 0,
    period: "1972-2023"
  },
  recovery: {
    compositeYear: 2024,
    retainedClasses: 13,
    totalClasses: 24,
    minPixels: 100,
    note: "NBR is spectral vegetation recovery, not biomass recovery."
  },
  carbon: [
    {
      source: "SCANFI v1.2",
      scope: "forested AOI",
      biomass: 68.44,
      carbon: 32.17,
      color: "#3b82f6",
      note: "Low mapped AGB estimate; live aboveground tree biomass."
    },
    {
      source: "MRNF Quebec",
      scope: "productive stands, 24.9% AOI",
      biomass: 102.97,
      carbon: 51.06,
      color: "#16a34a",
      note: "Provincial inventory-derived productive-stand reference; not ground truth."
    },
    {
      source: "ESA CCI",
      scope: "forested AOI / Method C",
      biomass: 171.93,
      carbon: 80.81,
      color: "#dc2626",
      note: "High mapped AGB estimate; explicit Hansen forest fraction."
    },
    {
      source: "SoilGrids",
      scope: "AOI, 0-30 cm",
      biomass: null,
      carbon: 59.8,
      color: "#7c3aed",
      note: "Soil organic carbon stock, 0-30 cm; separate pool."
    }
  ],
  commonFootprint: {
    areaHa: 2489.6,
    stands: 431,
    mrnfBiomass: 102.97,
    esaBiomass: 172.07,
    scanfiBiomass: 66.54,
    esaBias: 69.10,
    scanfiBias: -36.42,
    esaRmse: 72.39,
    scanfiRmse: 41.23,
    note: "On the identical MRNF productive-stand footprint, SCANFI has lower absolute bias and RMSE than ESA relative to MRNF."
  },
  biomassClasses: [
    { cls: "<50", area: 106.6, mrnf: 38.33, esa: 100.46, scanfi: 41.93 },
    { cls: "50-75", area: 160.3, mrnf: 65.27, esa: 125.20, scanfi: 52.28 },
    { cls: "75-100", area: 736.7, mrnf: 92.22, esa: 167.84, scanfi: 61.08 },
    { cls: "100-125", area: 1167.4, mrnf: 111.94, esa: 179.63, scanfi: 69.57 },
    { cls: ">=125", area: 318.7, mrnf: 135.50, esa: 192.16, scanfi: 84.29 }
  ]
};
