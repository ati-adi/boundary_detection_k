# Research 01 — Public Datasets for Agricultural Field Boundary Delineation

Date: 2026-07-26. Status: metadata research only; no large downloads performed.

Scope: public datasets of agricultural field/parcel boundaries (polygon vectors or
instance masks), typically paired with Sentinel-2 10 m imagery, relevant to
(a) training/benchmarking boundary models and (b) evaluating polygon quality of this
project's models (DelAny v1/v2, FTW PRUE B5, ensemble) on Kazakhstan-like
large-field steppe agriculture.

---

## 1. Fields of The World (FTW) Benchmark Dataset

- **Provider**: Kerner Lab (ASU), Microsoft AI for Good, Clark University, Taylor Geospatial, Radiant Earth. Org: https://github.com/fieldsoftheworld
- **Coverage**: 24–25 countries, 4 continents (Europe, Africa, Asia, South America).
- **Size**: ~70,462 samples, ~1.6 million harmonized field-boundary parcels. Each sample = multi-date Sentinel-2 chips (4 bands: B02/B03/B04/B08 @ 10 m, two temporal windows) + instance and semantic segmentation masks.
- **Format**: raster instance + semantic masks (GeoTIFF/COG) paired with S2 chips; harmonized vector sources underneath (fiboa standard).
- **Imagery**: Sentinel-2, 10 m, two seasonal composites per sample.
- **License**: mixed per-country (full set vs. "liberal" CC-BY-only subset both released).
- **Download**:
  - Source Cooperative: https://source.coop/kerner-lab/fields-of-the-world
  - Dataset used by `ftw-baselines` CLI: https://github.com/fieldsoftheworld/ftw-baselines (`ftw data download` after `pip install ftw-tools`; full dataset ~100 GB, but per-country partitions can be fetched individually).
  - HuggingFace hosts the **models**, not the dataset: `wherobots/prue-pt2` (the PRUE B5 weights this project uses), FTW U-Net weights in torchgeo.
- **KZ suitability**: **High for benchmarking** — it is the exact test distribution of the FTW PRUE B5 model, so it is the right ground truth for goal (3), verifying claimed quality on the model's own inputs. Includes some large-field countries (e.g. Ukraine-adjacent sources, South Africa commercial farms) but no Central Asia.
- **Small sample?** Yes — per-country subset download via source.coop / `ftw data download <country>`; a single country partition is a few hundred MB.
- Paper: https://arxiv.org/abs/2409.16252

## 2. FBIS-22M (Delineate Anything training set)

- **Provider**: Lavreniuk et al. (ESA / SRI NASU-SSAU / UMD). Project page: https://lavreniuk.github.io/Delineate-Anything/ ; code: https://github.com/Lavreniuk/Delineate-Anything
- **Coverage**: global multi-source (built from open cadastral/LPIS-style sources and manual annotation; underlying parcel sources span Europe, Africa, Asia, Americas).
- **Size**: 672,909 image patches, 22,926,427 instance masks — largest public field-boundary dataset.
- **Format**: raster instance masks paired with image patches.
- **Imagery**: multi-resolution 0.25 m–10 m (mixed aerial + satellite incl. Sentinel-2-like 10 m).
- **License**: released for research; check per-source terms on the project page.
- **Download**: links on the project page / GitHub README (large, multi-GB archives). The dataset is the training set of the DelAny v1/v2 weights this project downloads via `scripts/download_models.py`.
- **KZ suitability**: **Medium-High** — trained on very diverse field sizes incl. large Eastern-European fields, which is why DelAny transfers to KZ. As *evaluation* ground truth for KZ it is not directly usable (no KZ subset), but its test split is the right reference for goal (3) verification of DelAny claimed mAP (mAP@0.5 ~0.60+, +88.5% over prior SOTA per the paper).
- **Small sample?** Partially — the paper reports numbers on a held-out split; downloading the whole dataset is heavy. Metadata/figures freely available.
- Papers: https://arxiv.org/abs/2504.02534 (DelAny), https://arxiv.org/abs/2511.13417 (DelAnyFlow).

## 3. AI4Boundaries (EU JRC)

- **Provider**: European Commission JRC (d'Andrimont, Claverie, Waldner et al.). Paper: https://essd.copernicus.org/articles/15/317/2023/
- **Coverage**: 7 European regions — Austria, Catalonia (ES), France, Luxembourg, Netherlands, Slovenia, Sweden. ~47,105 km², ~2.5 M parcels, 7,831 stratified 4×4 km samples (stratified by perimeter/area ratio).
- **Format**: **both** — GSAA/LPIS parcel vector labels (GeoPackage) + raster masks.
- **Imagery**: Sentinel-2 L1C 10 m RGB-NIR monthly composites Mar–Aug 2019 (256×256 px chips) **plus 1 m aerial orthophotos** — unique pairing.
- **License**: open (EU JRC open data).
- **Download**: JRC FTP https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/ ; manifest `ai4boundaries_ftp_urls_all.csv` (train/val/test split), sampling `ai4boundaries_sampling.gpkg`. Helper package: `pip install git+https://github.com/waldnerf/ai4boundaries`.
- **KZ suitability**: **Medium** — EU small/medium parcels are smaller than KZ steppe fields, but it is the cleanest *vector + S2 + split* benchmark in existence; good for metric plumbing (object-level F1 vs `--ref` in this repo's `evaluate.py`).
- **Small sample?** **Yes — best-in-class for this.** Grab the CSV manifest + a handful of per-sample tiles directly over FTP/HTTP; tens of MB.

## 4. Radiant MLHub field-boundary / crop-type datasets

- **Provider**: Radiant Earth Foundation (MLHub STAC API, `https://api.radiant.earth/mlhub/v1`, requires free API token).
- Key collections:
  - **South Africa Crop Types Competition** (`ref_south_africa_crops_competition_v1`): ~3,800+ field polygons, Western Cape (large commercial wheat fields — closest African analogue to KZ field sizes), Sentinel-2 time series 2017, CC-BY-4.0. Used as boundary ground truth in the transfer-learning literature.
  - **Kenya PlantVillage Crop Type** (`ref_african_crops_kenya_02` / PlantVillage 2019): ~310–319 field boundaries surveyed on-foot in western Kenya, 242 km², S2 co-registered, CC-BY-SA-4.0. Smallholder — not KZ-like, test-only material.
  - **Uganda / Tanzania / Rwanda (NASA Harvest)**: similar smallholder crop-type-with-boundaries sets.
- **Format**: GeoJSON/vector labels + S2 chips via STAC.
- **KZ suitability**: South Africa subset **Medium** (large pivot/dryland fields); Kenya/Uganda **Low** (smallholder).
- **Small sample?** Yes — STAC API lets you pull label GeoJSON and a few chips only; `pip install radiant-mlhub`, e.g. `RadiantMLHubClient` → `client.list_collections()` / download archive per collection (label archives are a few MB).

## 5. France RPG / LPIS open data

- **Provider**: IGN / ASP (French paying agency). https://geoservices.ign.fr/rpg
- **Coverage**: all metropolitan France, ~9–10 M parcels/year, updated annually (RPG 2015–2024 vintages).
- **Format**: **vector polygons** (Shapefile/GeoPackage) with crop-type attributes. No imagery bundled — pair with your own S2.
- **License**: Licence Ouverte / Open Licence 2.0 (Etalab).
- **Download**: direct per-department/per-year zips from geoservices.ign.fr (a single département is 5–50 MB).
- **KZ suitability**: **Low-Medium** for GT (French fields much smaller), but it is the source behind PASTIS and many pretraining sets; useful for stress-testing polygon postprocessing/topology code.
- **Small sample?** Yes — one département download is trivial.

## 6. ZueriCrop

- **Provider**: ETH Zurich (Türkoglu et al., 2021).
- **Coverage**: canton of Zurich, Switzerland; ~116k parcels, 48 crop classes.
- **Format**: parcel polygons + Sentinel-2 **time series** (10 m, 2019) + Sentinel-1; aimed at crop-type classification, usable as boundary masks.
- **License**: research use; download: https://polybox.ethz.ch/index.php/s/nXfdr2AcXE3QNB6 (also mirrored in TorchGeo as `ZueriCrop`).
- **KZ suitability**: **Low** (small alpine fields, time-series format not polygon-first).
- **Small sample?** TorchGeo `torchgeo.datasets.ZueriCrop(..., download=True)` pulls the full archive (~3.9 GB) — moderate.

## 7. PASTIS / PASTIS-R

- **Provider**: Vivien Sainte Fare Garnot & Loic Landrieu (IGN/ENPC). https://github.com/VSainteuf/pastis-benchmark
- **Coverage**: 4 French regions, >4,000 km²; 2,433 patches of 128×128 px; 124,422 parcels, 18 crop classes.
- **Format**: raster semantic + **instance (panoptic) masks** + S2 time series (10 bands, 10 m, Sep 2018–Nov 2019, 38–61 dates); PASTIS-R adds Sentinel-1.
- **License**: open for research; download link in the GitHub README (~2–5 GB).
- **KZ suitability**: **Low** as KZ GT, **useful** as a panoptic-segmentation benchmark if instance-level metrics are wanted beyond FTW.
- **Small sample?** No official sample; full download only (but modest size).

## 8. Other notable datasets (quick hits)

- **AI4SmallFarms** (Vietnam & Cambodia, ~475k? — actually ~500 km² smallholder rice paddies, S2, ESSD 2023): smallholder SE-Asia; Low KZ relevance. https://doi.org/10.5194/essd-15-3991-2023
- **FTW Global products (PRUE predictions)**: country-scale **predicted** field-boundary maps 2024/2025 (1.6B polygons), on source.coop `ftw/global-data` — **not ground truth**, but a ready-made KZ polygon reference if Kazakhstan country extract is released (currently Japan, Mexico, Rwanda, South Africa, Switzerland are public).
- **EuroCrops** (TUM): harmonized LPIS crop-type polygons for ~16 EU countries; boundaries usable, no imagery. https://github.com/maja601/EuroCrops
- **AgriFieldNet / CV4A Kenya (Zindi)**, **CropHarvest**: field-level labels, smallholder Africa; Low KZ relevance.
- **Denethor / AgriSen-COG / TimeSen2Crop**: crop-type chips, not boundary GT.
- **Global 10 m field-boundary map** (2026, arXiv 2605.11055): first global 10 m field boundary product — watch for KZ availability as a cross-check reference.

---

## Comparison table

| Dataset | Provider | Coverage | Fields/labels | Format | Imagery | License | Download | KZ-like GT? | Small sample? |
|---|---|---|---|---|---|---|---|---|---|
| FTW benchmark | Kerner Lab/MS/Radiant | 24–25 countries, 4 continents | 1.6 M parcels / 70,462 samples | instance+semantic masks + S2 chips | S2 10 m, 2 seasons | mixed / CC-BY subset | source.coop/kerner-lab/fields-of-the-world; `ftw data download` | **High for model verification (PRUE's own test set)**; no Central Asia | Yes (per-country) |
| FBIS-22M | Lavreniuk et al. | global multi-source | 22.9 M instances / 672,909 patches | instance masks + patches | 0.25–10 m mixed | research | lavreniuk.github.io/Delineate-Anything | Medium-High (DelAny's own test set) | No real sample |
| AI4Boundaries | EU JRC | 7 EU regions, 47k km² | 2.5 M parcels / 7,831 samples | **vectors (GPKG) + masks** | S2 10 m 2019 + 1 m aerial | open (EU) | jeodpp.jrc.ec.europa.eu FTP + manifest CSV | Medium | **Yes (per-sample tiles)** |
| MLHub South Africa | Radiant Earth | Western Cape, ZA | ~3.8k polygons | GeoJSON + S2 chips | S2 2017 TS | CC-BY-4.0 | api.radiant.earth/mlhub (STAC, token) | Medium (large dryland fields) | Yes (labels only, MBs) |
| MLHub Kenya (PlantVillage) | PlantVillage/Radiant | W. Kenya, 242 km² | ~310 polygons | GeoJSON + S2 chips | S2 2019 TS | CC-BY-SA-4.0 | same STAC API | Low (smallholder) | Yes |
| France RPG | IGN/ASP | France, yearly | ~9–10 M parcels/yr | **vectors (SHP/GPKG)** | none (pair w/ S2) | Licence Ouverte 2.0 | geoservices.ign.fr/rpg | Low-Medium | Yes (per département) |
| ZueriCrop | ETH Zurich | Zurich canton | ~116k parcels | polygons + TS rasters | S2 10 m 2019 TS + S1 | research | polybox.ethz.ch / TorchGeo | Low | ~3.9 GB full |
| PASTIS(-R) | Garnot & Landrieu | 4 French regions, 4k km² | 124k parcels / 2,433 patches | panoptic masks + TS | S2 10 m TS (+S1) | research | github.com/VSainteuf/pastis-benchmark | Low | No |
| AI4SmallFarms | (ESSD 2023) | Vietnam, Cambodia | smallholder paddies | vectors + masks | S2 10 m | open | Zenodo via paper | Low | Yes |

## Sample-download cheat sheet (metadata/small only)

```bash
# AI4Boundaries manifest + sampling geopackage (tiny):
curl -O https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/ai4boundaries_ftp_urls_all.csv
curl -O https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/ai4boundaries_sampling.gpkg
#   then pick individual sample tiles from the URLs in the CSV.

# FTW benchmark (per-country subset, after env setup):
pip install ftw-tools
ftw data download --help        # lists countries; download one country only

# Radiant MLHub labels (needs free token at mlhub.earth):
pip install radiant-mlhub
#   collections: ref_south_africa_crops_competition_v1, ref_african_crops_kenya_02

# France RPG (one département, e.g. 2023):
#   browse https://geoservices.ign.fr/rpg → direct zip link

# ZueriCrop / PASTIS via TorchGeo (full, moderate):
python -c "from torchgeo.datasets import ZueriCrop; ZueriCrop(root='data', download=True)"
```

## Recommendation for this project

1. **Goal (3) — verify claimed model quality on the models' own inputs**: use the **FTW benchmark test split** for FTW PRUE B5 (its exact benchmark; PRUE paper reports SOTA on it) and the **FBIS-22M test split** (or the paper's reported mAP) for DelAny. FTW per-country download is the practical path.
2. **Goal (4) — KZ polygon-quality assessment**: there is **no public KZ field-boundary ground truth**. Best proxies: (a) AI4Boundaries vectors + S2 to build/validate the object-level evaluation code (`scripts/evaluate.py --ref`), (b) MLHub South Africa for large-field comparisons, (c) hand-digitized reference polygons over 2–3 KZ AOIs (the realistic option, supported by the `--ref` flag), cross-checked against the FTW global PRUE country predictions if/when a KZ extract ships.
