# 船海平台硬编码物性 -> TensorDG 物性库迁移映射表

**生成方式**: `python scripts/props_migration_inventory.py <ship repo>`（只读扫描，未修改 ship 仓任何文件；本表由程序生成，无手抄数字）

**扫描口径**: `*.py` 中以字面量赋值的 `rho*/density*/nu*/mu*/viscos*/young*/poisson*/yield*/strength*/manning*` 常量。

**统计**: conflict=12, migrate=800, review=453, total=1265

| action | 含义 |
|---|---|
| migrate | 量纲折算后与库值相差 ≤10%，可改为 `props.load_db().get(...)` |
| conflict | 相差 >10%，需 owner 判定（可能是别的物质/温度/单位） |
| review | 无量纲系数、单位不明或物质未标注，**不并入**物性库 |

> 诚实声明：扫描器只能看到标识符与字面量，看不到单位与物质。单位由量级推断（kg/m³ vs t/m³、Pa vs MPa），推断结果写在 “口径” 列；`migrate` 只是候选，仍需模块 owner 确认工作介质。

## 逐条映射表

| # | file:line | identifier | 现值 | action | 库条目 | 库值(SI) | 相对偏差 | 口径/原因 | 原注释 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `artifacts/EX1_tank_test_spec/make_audit_figs.py:24` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 2 | `backend/api/abkowitz.py:135` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 3 | `backend/api/added_resistance.py:104` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 4 | `backend/api/air_resistance.py:25` | `RHO_AIR_DEFAULT` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³, ISO 标准大气 (15°C, 101325 Pa) |
| 5 | `backend/api/air_resistance.py:26` | `RHO_AIR_15C` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 6 | `backend/api/air_resistance.py:27` | `RHO_AIR_20C` | 1.204 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 7 | `backend/api/bem_radiation_diffraction/bem_core.py:53` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³  seawater density |
| 8 | `backend/api/buckling_bc_maturity.py:419` | `rho` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 9 | `backend/api/buckling_bc_maturity.py:517` | `nu_al` | 0.33 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 10 | `backend/api/buckling_bc_maturity.py:522` | `nu_st` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 11 | `backend/api/cabin_noise.py:27` | `RHO_AIR` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 [kg/m³] (20°C) |
| 12 | `backend/api/cabin_noise.py:29` | `RHO_AIR_REF` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 参考空气密度 |
| 13 | `backend/api/cargo_loading.py:58` | `RHO_SEA` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m^3, seawater density |
| 14 | `backend/api/cargo_selfheating.py:410` | `rho_air` | 1.2 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 15 | `backend/api/cavitation.py:81` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 16 | `backend/api/cavitation_detailed.py:32` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 17 | `backend/api/cavitation_inception.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ (海水) |
| 18 | `backend/api/cavitation_inception.py:46` | `RHO_FW` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ (淡水, 空泡筒) |
| 19 | `backend/api/cavitation_inception.py:51` | `NU_FW_15` | 1.139e-06 | migrate | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 0.0% | SI | m²/s (淡水 15°C 运动粘度) |
| 20 | `backend/api/cavitation_inception.py:52` | `NU_FW_20` | 1.002e-06 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 12.0% | SI | m²/s (淡水 20°C) |
| 21 | `backend/api/cavitation_inception.py:53` | `NU_SEA_15` | 1.189e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s (海水 15°C) |
| 22 | `backend/api/cavitation_inception_envelope.py:39` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 (15°C) |
| 23 | `backend/api/cavitation_inception_envelope.py:40` | `RHO_FRESH` | 998.2 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ 淡水密度 (15°C) |
| 24 | `backend/api/cavitation_inception_envelope.py:41` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s 海水运动粘度 (15°C) |
| 25 | `backend/api/cavitation_inception_envelope.py:42` | `NU_FRESH` | 1.14e-06 | migrate | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 0.1% | SI | m²/s 淡水运动粘度 (15°C) |
| 26 | `backend/api/confined_water_25d.py:45` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 [kg/m³] |
| 27 | `backend/api/coupled_maneuvering.py:27` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 28 | `backend/api/coupled_maneuvering.py:28` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 29 | `backend/api/coupled_maneuvering.py:29` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 30 | `backend/api/damage_stability_full.py:215` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 31 | `backend/api/damage_stability_full.py:486` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 32 | `backend/api/docking_loads.py:62` | `RHO_SW` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 33 | `backend/api/draught_survey.py:104` | `rho_actual` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | Actual water density [t/m³] |
| 34 | `backend/api/draught_survey.py:105` | `rho_standard` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | Standard seawater density [t/m³] |
| 35 | `backend/api/draught_survey.py:109` | `u_density` | 0.002 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | Density uncertainty [t/m³] |
| 36 | `backend/api/drydocking_hull.py:35` | `WATER_RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 37 | `backend/api/dynamic_stability.py:38` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 kg/m³（动压口径用） |
| 38 | `backend/api/electric_propulsion_noise.py:593` | `rho_fluid` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 密度 [kg/m^3] |
| 39 | `backend/api/electric_propulsion_noise.py:596` | `rho_fluid` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 40 | `backend/api/fea.py:62` | `yield_stress` | 235 | review | `-` | - | - | yield stress, steel grade not identified in the code | MPa |
| 41 | `backend/api/fea.py:94` | `yield_stress` | 235 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 42 | `backend/api/flettner_rotor.py:37` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³, 标准大气海平面空气密度 |
| 43 | `backend/api/flettner_rotor.py:38` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³, 海水密度 |
| 44 | `backend/api/flight_deck_envelope.py:41` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 45 | `backend/api/grain_stability.py:63` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 46 | `backend/api/green_shipping.py:579` | `AIR_DENSITY_KG_M3` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 47 | `backend/api/green_shipping_extended.py:230` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 48 | `backend/api/green_shipping_extended.py:320` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 49 | `backend/api/gust_admittance.py:33` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 kg/m³ |
| 50 | `backend/api/gust_admittance.py:34` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 kg/m³ |
| 51 | `backend/api/ice/astern_icebreaking.py:75` | `rho_ice` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 52 | `backend/api/ice/astern_icebreaking.py:76` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 53 | `backend/api/ice/astern_icebreaking.py:406` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 54 | `backend/api/ice/astern_icebreaking.py:466` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 55 | `backend/api/ice/broken_channel/in22_dem_data.py:56` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 56 | `backend/api/ice/broken_channel/integration.py:22` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 57 | `backend/api/ice/formulas.py:67` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | 弯曲强度 σf (Pa), 典型 300-700 kPa |
| 58 | `backend/api/ice/formulas.py:68` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 ρi (kg/m³) |
| 59 | `backend/api/ice/formulas.py:69` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 ρw (kg/m³) |
| 60 | `backend/api/ice/ice_verification.py:52` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 61 | `backend/api/ice/ice_verification.py:53` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 62 | `backend/api/ice/ice_verification.py:181` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 63 | `backend/api/ice/ice_verification.py:182` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 64 | `backend/api/ice/ice_verification.py:252` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 65 | `backend/api/ice/ice_verification.py:391` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 66 | `backend/api/ice/ice_verification.py:457` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 67 | `backend/api/ice/ice_verification.py:458` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 68 | `backend/api/ice/maturity.py:24` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 |
| 69 | `backend/api/ice/ridge_ice.py:62` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 (kg/m³) |
| 70 | `backend/api/ice/ridge_ice.py:63` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 71 | `backend/api/ice/sample_bank.py:42` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified |  |
| 72 | `backend/api/ice/sample_bank.py:43` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 73 | `backend/api/ice/sample_bank.py:44` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 74 | `backend/api/ice/xval/anchor_data.py:192` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3, sea-ice density at the benchmark temperatu |
| 75 | `backend/api/ice/xval/anchor_data.py:193` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 76 | `backend/api/ice_coupled_simulation.py:29` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 77 | `backend/api/ice_coupled_simulation.py:30` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 78 | `backend/api/ice_fatigue.py:32` | `ice_flexural_strength_pa` | 500000 | review | `-` | - | - | unclassified |  |
| 79 | `backend/api/ice_fatigue.py:45` | `young_modulus_pa` | 2.069e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 80 | `backend/api/ice_fatigue.py:46` | `yield_strength_pa` | 3.55e+08 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 81 | `backend/api/ice_fatigue.py:62` | `ice_flexural_strength_pa` | 500000 | review | `-` | - | - | unclassified |  |
| 82 | `backend/api/ice_fatigue.py:73` | `young_modulus_pa` | 2.069e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 83 | `backend/api/ice_fatigue.py:74` | `yield_strength_pa` | 3.55e+08 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 84 | `backend/api/ice_maneuverability_coupling.py:36` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 85 | `backend/api/ice_maneuverability_coupling.py:37` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 86 | `backend/api/ice_maneuverability_coupling.py:38` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 87 | `backend/api/ice_maneuvering_verification.py:34` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 88 | `backend/api/ice_maneuvering_verification.py:35` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 89 | `backend/api/ice_maneuvering_verification.py:87` | `nu_ice` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 90 | `backend/api/ice_navigation.py:36` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified |  |
| 91 | `backend/api/ice_navigation.py:37` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 92 | `backend/api/ice_navigation.py:38` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 93 | `backend/api/ice_rao.py:80` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 冰密度 kg/m³ |
| 94 | `backend/api/ice_rao.py:121` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 95 | `backend/api/ice_stability.py:56` | `ICE_DENSITY_DEFAULT` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m³ (fresh ice, IMO default) |
| 96 | `backend/api/ice_stability.py:57` | `ICE_DENSITY_BRACKISH` | 1000 | conflict | `sea_ice/density` | 900 kg/m^3 | 11.1% | SI; compared at T=0 degC (domain boundary) | kg/m³ (brackish/salt ice) |
| 97 | `backend/api/ice_stability.py:60` | `WATER_DENSITY_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 98 | `backend/api/ice_stability.py:84` | `rho_liquid` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | Liquid density (kg/m³) |
| 99 | `backend/api/ice_uq.py:240` | `rho_t_per_m3` | 1.025 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) |  |
| 100 | `backend/api/ikeda_roll_damping.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 101 | `backend/api/ikeda_roll_damping.py:47` | `MU_SEA` | 0.00108 | conflict | `sea_water/viscosity_dynamic` | 0.00121908 Pa*s | 11.4% | SI | Pa·s 海水动力黏度 (15°C) |
| 102 | `backend/api/imo_is_code.py:77` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ —— 海平面标准空气密度（OCIMF 动压口径） |
| 103 | `backend/api/imo_is_code.py:78` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ —— 海水密度 |
| 104 | `backend/api/inclining_experiment.py:70` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 105 | `backend/api/inclining_experiment.py:71` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ |
| 106 | `backend/api/inclining_test.py:153` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | Water density (t/m³) |
| 107 | `backend/api/irregular_wave_added_resistance.py:65` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] |
| 108 | `backend/api/kvlcc2_mmg_maneuvering.py:22` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 109 | `backend/api/lbm.py:1406` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 110 | `backend/api/lifting_line.py:221` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 111 | `backend/api/lifting_theory.py:253` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 |
| 112 | `backend/api/lifting_theory.py:294` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 113 | `backend/api/loading_instrument.py:39` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 114 | `backend/api/loading_stability.py:41` | `RHO_SEA` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m³ 海水密度 |
| 115 | `backend/api/loading_stability.py:69` | `cargo_density` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 116 | `backend/api/long_term_wave_load.py:50` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 117 | `backend/api/maneuvering_astern.py:70` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m^3 |
| 118 | `backend/api/maneuvering_astern.py:232` | `mu_ice` | 0.12 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-钢摩擦系数 |
| 119 | `backend/api/maneuvering_astern.py:266` | `ice_flexural_strength` | 500000 | review | `-` | - | - | unclassified |  |
| 120 | `backend/api/maneuvering_ice.py:48` | `ice_flexural_strength` | 500000 | review | `-` | - | - | unclassified | 冰弯曲强度 (Pa) |
| 121 | `backend/api/maneuvering_ice.py:49` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 122 | `backend/api/maneuvering_ice.py:65` | `ice_flexural_strength` | 500000 | review | `-` | - | - | unclassified |  |
| 123 | `backend/api/maneuvering_ice.py:66` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 124 | `backend/api/maneuvering_ice.py:83` | `ice_flexural_strength` | 500000 | review | `-` | - | - | unclassified |  |
| 125 | `backend/api/maneuvering_ice.py:84` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 126 | `backend/api/maneuvering_mmg.py:155` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 127 | `backend/api/maneuvering_mmg.py:493` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 128 | `backend/api/marine_ice_accretion.py:32` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 冰密度 kg/m³（淡水冰） |
| 129 | `backend/api/marine_ice_accretion.py:33` | `RHO_ICE_SPRAY` | 890 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | 飞沫冰密度 kg/m³（含卤水气泡） |
| 130 | `backend/api/marine_icing.py:55` | `RHO_ICE_DEFAULT` | 890 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | 飞沫冰密度 kg/m^3（含卤水气泡） |
| 131 | `backend/api/marine_icing.py:93` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 132 | `backend/api/mmg_model.py:54` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 133 | `backend/api/model_test_data_generator.py:187` | `water_density` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 淡水 |
| 134 | `backend/api/model_test_data_generator.py:189` | `water_viscosity` | 1e-06 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 12.2% | SI |  |
| 135 | `backend/api/moonpool.py:53` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 136 | `backend/api/mooring_dynamics.py:36` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ (seawater) |
| 137 | `backend/api/mooring_dynamics.py:37` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 138 | `backend/api/nonlinear_roll.py:43` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 139 | `backend/api/parametric_roll.py:78` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | seawater density kg/m³ |
| 140 | `backend/api/pinn.py:43` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 141 | `backend/api/podded_ice_maneuvering.py:57` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m^3 |
| 142 | `backend/api/podded_ice_maneuvering.py:112` | `mu_ice` | 0.12 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-钢摩擦系数 |
| 143 | `backend/api/potential_3d.py:336` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 144 | `backend/api/potential_3d.py:454` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 145 | `backend/api/potential_flow_3d.py:246` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 146 | `backend/api/potential_flow_3d.py:260` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 147 | `backend/api/propeller_design.py:38` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 148 | `backend/api/propeller_design.py:80` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 149 | `backend/api/propeller_hull_interaction.py:54` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 150 | `backend/api/propeller_openwater.py:38` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 151 | `backend/api/propeller_pressure.py:26` | `_RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 152 | `backend/api/propeller_rudder_interaction.py:49` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 153 | `backend/api/propeller_rudder_interaction.py:334` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 |
| 154 | `backend/api/propeller_rudder_interaction.py:522` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 155 | `backend/api/propeller_vibration.py:52` | `_NU_STEEL` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 钢泊松比 |
| 156 | `backend/api/propeller_vibration.py:53` | `_RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | 钢密度 kg/m³ |
| 157 | `backend/api/propulsion.py:55` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 158 | `backend/api/propulsion.py:275` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 159 | `backend/api/propulsion_core.py:27` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水密度 |
| 160 | `backend/api/resistance.py:220` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 161 | `backend/api/resistance_advanced.py:58` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 162 | `backend/api/resistance_advanced.py:59` | `water_viscosity` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 163 | `backend/api/resistance_advanced.py:99` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 164 | `backend/api/resistance_advanced.py:100` | `water_viscosity` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 165 | `backend/api/resistance_advanced.py:132` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 166 | `backend/api/resistance_cfd_benchmark.py:37` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 167 | `backend/api/resistance_cfd_benchmark.py:38` | `RHO_FRESH` | 998.2 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m^3 (20°C) |
| 168 | `backend/api/resistance_cfd_benchmark.py:39` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s (15°C) |
| 169 | `backend/api/resistance_cfd_benchmark.py:40` | `NU_FRESH_20C` | 1.004e-06 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 11.8% | SI | m^2/s (20°C) |
| 170 | `backend/api/resistance_p1.py:162` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 171 | `backend/api/resistance_p1.py:229` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 172 | `backend/api/resistance_p1.py:296` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 173 | `backend/api/restricted_wave_maneuvering.py:41` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 174 | `backend/api/roll_damping.py:56` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 175 | `backend/api/roughness.py:49` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 176 | `backend/api/roughness.py:50` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 海水运动粘度 m²/s |
| 177 | `backend/api/seakeeping.py:52` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 178 | `backend/api/seakeeping.py:227` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 179 | `backend/api/seakeeping.py:388` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 180 | `backend/api/seakeeping_advanced.py:45` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 181 | `backend/api/seakeeping_rao_full.py:40` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 182 | `backend/api/seakeeping_stf_added_resistance.py:60` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 183 | `backend/api/seakeeping_stf_added_resistance.py:61` | `RHO_FRESH` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | Wigley III 实验在淡水中 |
| 184 | `backend/api/second_order_drift.py:59` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 185 | `backend/api/series60_resistance.py:43` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 186 | `backend/api/series60_resistance.py:44` | `water_viscosity` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 187 | `backend/api/shaft_alignment.py:46` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 188 | `backend/api/shallow_water_maneuvering.py:359` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 189 | `backend/api/shallow_water_two_ship.py:48` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 190 | `backend/api/ship_vibration.py:30` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 191 | `backend/api/slamming_whipping.py:60` | `rho_steel` | 7850 | review | `-` | - | - | density of an unlabelled substance | 钢密度 [kg/m³] |
| 192 | `backend/api/spectral_fatigue.py:41` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 193 | `backend/api/stawave2_added_resistance.py:69` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 194 | `backend/api/structure.py:35` | `yield_stress` | 235 | review | `-` | - | - | yield stress, steel grade not identified in the code | 屈服应力 (MPa) |
| 195 | `backend/api/torsional_vibration.py:23` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 196 | `backend/api/waterjet.py:30` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 197 | `backend/api/waterjet.py:31` | `RHO_FRESH` | 999 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.0% | SI | kg/m³ 淡水密度 |
| 198 | `backend/api/wave_drift_force.py:34` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 199 | `backend/api/wave_energy_wec.py:48` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 200 | `backend/api/wave_maneuvering/wave_manueuvering.py:60` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 201 | `backend/api/wave_maneuvering.py:27` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 202 | `backend/api/wave_speed_loss.py:58` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] |
| 203 | `backend/api/wave_speed_loss.py:59` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | [kg/m³] |
| 204 | `backend/api/wet_modal_api.py:24` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 205 | `backend/api/wet_modal_api.py:117` | `rho_water` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 水密度→0 |
| 206 | `backend/api/wet_modal_api.py:137` | `rho_water` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 207 | `backend/geometry/hydro_curves.py:65` | `RHO_SEAWATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m³ |
| 208 | `backend/icing/ice_accretion.py:36` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m³  淡水冰密度 |
| 209 | `backend/icing/ice_accretion.py:37` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³  空气密度 (0°C) |
| 210 | `backend/kb_samples/spec.py:47` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 211 | `backend/kb_samples/spec.py:48` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 212 | `backend/knowledge/anchors.py:28` | `RHO` | 1025.5 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.0% | SI |  |
| 213 | `backend/knowledge/generators.py:25` | `NU_SEAWATER` | 1.1883e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.0% | SI | m²/s |
| 214 | `backend/knowledge/generators.py:26` | `RHO_SEAWATER` | 1025.5 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.0% | SI | kg/m³ |
| 215 | `backend/methods/crp/__init__.py:60` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | Water density [kg/m³] |
| 216 | `backend/methods/manoeuvring/kvlcc2_circle.py:75` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 217 | `backend/methods/manoeuvring/model.py:71` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 218 | `backend/methods/manoeuvring/restricted_water.py:54` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 219 | `backend/methods/manoeuvring/single_shaft_failure.py:95` | `KINEMATIC_VISCOSITY` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s, ITTC-78 fresh/sea water at 15 C |
| 220 | `backend/methods/manoeuvring/submersible.py:61` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, ITTC/ISO seawater density used platform-wi |
| 221 | `backend/methods/restricted_waterway/__init__.py:36` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 222 | `backend/methods/roll_damping/ikeda.py:50` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 223 | `backend/methods/roll_damping/ikeda.py:52` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 海水运动黏度 m²/s (15°C) |
| 224 | `backend/methods/squat_core.py:48` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 225 | `backend/methods/squat_core.py:49` | `RHO_RIVER` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 淡水密度 [kg/m³] |
| 226 | `backend/methods/structural/wet_modal.py:51` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 227 | `backend/methods/structural/wet_modal.py:73` | `rho_struct` | 7850 | review | `-` | - | - | density of an unlabelled substance | 结构密度 kg/m³ |
| 228 | `backend/methods/wind_load_ocimf.py:40` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³, 标准空气密度 |
| 229 | `backend/ml/conservation_laws.py:21` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ (15°C) |
| 230 | `backend/ml/kvlcc2_data.py:24` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 231 | `backend/ml/kvlcc2_data.py:26` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 232 | `backend/ml/pinn_resistance.py:21` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 233 | `backend/ml/pinn_resistance.py:23` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 234 | `backend/modules/added_resistance/raw_empirical.py:28` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 235 | `backend/modules/added_resistance/raw_empirical.py:29` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m^3 |
| 236 | `backend/modules/added_resistance/tests/test_added_resistance_maturity.py:58` | `RHO_REF` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 237 | `backend/modules/air_layer_drag_reduction/air_layer_model.py:35` | `RHO_SW` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³  seawater density at 15°C |
| 238 | `backend/modules/air_layer_drag_reduction/air_layer_model.py:37` | `NU_SW` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s   kinematic viscosity at 15°C |
| 239 | `backend/modules/air_layer_drag_reduction/air_layer_model.py:38` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³  air density at sea level |
| 240 | `backend/modules/appendage_resistance/taxonomy.py:35` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 15 degC |
| 241 | `backend/modules/appendage_resistance/taxonomy.py:36` | `NU_SEAWATER` | 1.1883e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.0% | SI | m^2/s, 15 degC |
| 242 | `backend/modules/astern_ice/astern_ice_resistance.py:55` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 243 | `backend/modules/astern_ice/astern_ice_resistance.py:56` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 [kg/m³] |
| 244 | `backend/modules/astern_ice/four_quadrant.py:75` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 245 | `backend/modules/astern_ice/rudder_degradation.py:54` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 246 | `backend/modules/astern_icebreaking/astern_ice_resistance.py:111` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | 弯曲强度 σf (Pa) |
| 247 | `backend/modules/astern_icebreaking/astern_ice_resistance.py:112` | `compressive_strength` | 2e+06 | review | `-` | - | - | unclassified | 抗压强度 σc (Pa) |
| 248 | `backend/modules/astern_icebreaking/astern_ice_resistance.py:113` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 ρi (kg/m³) |
| 249 | `backend/modules/astern_icebreaking/astern_ice_resistance.py:114` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 ρw (kg/m³) |
| 250 | `backend/modules/astern_icebreaking/four_quadrant_propeller.py:101` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 251 | `backend/modules/astern_icebreaking/four_quadrant_propeller.py:322` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 252 | `backend/modules/atmospheric_icing/makkonen.py:37` | `RHO_WATER` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 253 | `backend/modules/atmospheric_icing/makkonen.py:38` | `RHO_ICE_DEFAULT` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 工程积冰密度 kg/m^3（含气泡/卤水 850-920） |
| 254 | `backend/modules/atmospheric_icing.py:67` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 冰密度 kg/m³（大气结冰典型值 800-950） |
| 255 | `backend/modules/atmospheric_icing.py:68` | `RHO_WATER` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 水密度 kg/m³ |
| 256 | `backend/modules/ballast_water/pump.py:46` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 257 | `backend/modules/ballast_water/tankage.py:48` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 258 | `backend/modules/bayesian_scale_calibration/data.py:57` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 259 | `backend/modules/bayesian_scale_calibration/data.py:58` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 260 | `backend/modules/bayesian_scale_calibration/scale_law.py:132` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 261 | `backend/modules/berthing/mmg_dynamics.py:21` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 262 | `backend/modules/berthing/mmg_dynamics.py:22` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 263 | `backend/modules/brash_ice/dem.py:43` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 264 | `backend/modules/brash_ice/dem.py:44` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 265 | `backend/modules/brash_ice/dem.py:76` | `mu_ii` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 266 | `backend/modules/brash_ice/dem.py:77` | `mu_ih` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 267 | `backend/modules/brash_ice/dem.py:78` | `mu_wall` | 0.2 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 268 | `backend/modules/brash_ice/dem.py:79` | `mu_b` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 269 | `backend/modules/brash_ice_stability/geometry.py:21` | `RHO_WATER_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水（可与淡水 1000 切换） |
| 270 | `backend/modules/brash_ice_stability/geometry.py:22` | `RHO_ICE_DEFAULT` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3 海冰（0 ℃ 附近，~900-930 文献区间） |
| 271 | `backend/modules/broken_ice_resistance/field.py:36` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3, 海冰密度（无盐度修正的常用工程值） |
| 272 | `backend/modules/broken_ice_resistance/field.py:37` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水密度 |
| 273 | `backend/modules/broken_ice_resistance/openwater.py:30` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 274 | `backend/modules/broken_ice_resistance/params.py:64` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 275 | `backend/modules/cabin_noise/finite_panel.py:85` | `RHO0` | 1.2 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 276 | `backend/modules/cabin_noise/standards.py:56` | `RHO_AIR` | 1.204 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3  (rho*c = 413; 工程常用 400/415) |
| 277 | `backend/modules/damage_stability_3dof/equilibrium_solver.py:107` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 278 | `backend/modules/damage_stability_3dof/gz_curve_trim.py:162` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 279 | `backend/modules/damaged_stability/core.py:68` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 280 | `backend/modules/deck_water/free_surface.py:67` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 281 | `backend/modules/deck_water/spectral_green.py:68` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 282 | `backend/modules/deck_water/swe.py:41` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 283 | `backend/modules/electric_propulsion/chain.py:39` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 284 | `backend/modules/electric_propulsion/stages.py:333` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 285 | `backend/modules/evidence_upgrade.py:99` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水 |
| 286 | `backend/modules/evidence_upgrade.py:100` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 空气 |
| 287 | `backend/modules/fire_smoke_zone_model.py:45` | `fire_soot_yield` | 0.05 | review | `-` | - | - | unclassified | kg/kg (soot yield) |
| 288 | `backend/modules/flooding/acceptance.py:24` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 289 | `backend/modules/flooding/simulate.py:79` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 290 | `backend/modules/fouling_cleaning/correlations.py:47` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m^3] |
| 291 | `backend/modules/fouling_cleaning/correlations.py:48` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 15 ℃ 海水运动粘性 [m^2/s] |
| 292 | `backend/modules/fouling_dv_identifiability/physics.py:29` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 293 | `backend/modules/fouling_dv_identifiability/physics.py:35` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 294 | `backend/modules/fouling_dv_identifiability/physics.py:274` | `rho_ar` | 0.6 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | AR(1) 系数 |
| 295 | `backend/modules/frazil_icing/capture.py:84` | `rho` | 930 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | 含孔隙密度 [kg/m3] |
| 296 | `backend/modules/frazil_icing/layer.py:63` | `rho_layer` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 沉积层密度（含孔隙，比纯冰略低）[kg/m3] |
| 297 | `backend/modules/frazil_icing/roughness.py:32` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 298 | `backend/modules/frazil_icing/stability.py:30` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 299 | `backend/modules/frazil_icing/thermal.py:36` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 纯冰密度 [kg/m3]（frazil 团絮含孔隙，见 capture.FrazilParticle |
| 300 | `backend/modules/frontier_integration/cylinder_icing.py:65` | `RHO_A` | 1.29 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 kg/m^3 |
| 301 | `backend/modules/frontier_integration/cylinder_icing.py:66` | `MU_A` | 1.75e-05 | review | `-` | - | - | viscosity of an unlabelled substance | 空气动力粘度 Pa s |
| 302 | `backend/modules/frontier_integration/moonpool.py:55` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 303 | `backend/modules/frontier_integration/weather_routing.py:68` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 304 | `backend/modules/fuel_processing/centrifugal.py:269` | `hfo_density_kgm3` | 980 | review | `-` | - | - | density of an unlabelled substance |  |
| 305 | `backend/modules/fuel_processing/centrifugal.py:300` | `hfo_density_kgm3` | 980 | review | `-` | - | - | density of an unlabelled substance |  |
| 306 | `backend/modules/fuel_processing/chain.py:345` | `nu_injection_cSt` | 14 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 307 | `backend/modules/fuel_processing/chain.py:363` | `nu_injection_cSt` | 12 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 308 | `backend/modules/fuel_processing/chain.py:381` | `nu_injection_cSt` | 6 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 309 | `backend/modules/full_scale_correlation.py:75` | `RHO_SEA_15C` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 310 | `backend/modules/full_scale_correlation.py:76` | `NU_SEA_15C` | 1.1883e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.0% | SI |  |
| 311 | `backend/modules/full_scale_correlation.py:77` | `RHO_FRESH_15C` | 999.1 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.0% | SI |  |
| 312 | `backend/modules/full_scale_correlation.py:78` | `NU_FRESH_15C` | 1.1386e-06 | migrate | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 0.0% | SI |  |
| 313 | `backend/modules/gpu_lbm_solver.py:236` | `nu_t` | 0 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) |  |
| 314 | `backend/modules/green_navigation_ice/chain.py:290` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 315 | `backend/modules/green_navigation_ice/feasible_domain.py:58` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 316 | `backend/modules/green_navigation_ice/feasible_domain.py:565` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 317 | `backend/modules/green_navigation_ice/ice_state.py:55` | `T_REF_STRENGTH` | -10 | review | `-` | - | - | unclassified | ℃ |
| 318 | `backend/modules/green_navigation_ice/ice_state.py:59` | `K_MU_TEMP` | 0.012 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) | 1/℃ |
| 319 | `backend/modules/green_navigation_ice/ice_state.py:310` | `rho_hc` | 0.45 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 320 | `backend/modules/green_navigation_ice/ice_state.py:311` | `rho_ht` | -0.55 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 321 | `backend/modules/green_navigation_ice/ice_state.py:312` | `rho_ct` | -0.25 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 322 | `backend/modules/green_navigation_ice/w3n11_physics.py:46` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 323 | `backend/modules/green_navigation_ice/w3n11_physics.py:47` | `RHO_SNOW` | 300 | review | `-` | - | - | density of an unlabelled substance |  |
| 324 | `backend/modules/green_navigation_ice/w3n11_physics.py:48` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 325 | `backend/modules/green_navigation_ice/w3n11_physics.py:50` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 |
| 326 | `backend/modules/green_navigation_ice/w3n11_physics.py:52` | `MU_V_REF` | 0.6 | review | `-` | - | - | viscosity of an unlabelled substance | 摩擦速度软化参考速度 [m/s] |
| 327 | `backend/modules/green_navigation_ice/w3n11_physics.py:53` | `MU_MIN_RATIO` | 0.55 | review | `-` | - | - | dimensionless/derived coefficient | 高速极限摩擦系数 / μ0 |
| 328 | `backend/modules/green_navigation_ice/w3n11_physics.py:141` | `MU_V_REF_SAMPLE` | 1 | review | `-` | - | - | viscosity of an unlabelled substance | 取摩擦系数的参考滑移速度 [m/s] |
| 329 | `backend/modules/green_navigation_ice/w3n11_vessels.py:17` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水 |
| 330 | `backend/modules/green_propulsion/bearing_lubrication.py:66` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 331 | `backend/modules/green_propulsion/bearing_lubrication.py:110` | `oil_viscosity` | 0.1 | review | `-` | - | - | viscosity of an unlabelled substance | Pa·s |
| 332 | `backend/modules/green_propulsion/spp_propulsor.py:53` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³, 海水密度 |
| 333 | `backend/modules/green_propulsion/spp_propulsor.py:54` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³, 淡水密度 |
| 334 | `backend/modules/green_shipping_iso19030/ship_model.py:36` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水（15 ℃） |
| 335 | `backend/modules/green_shipping_iso19030/ship_model.py:37` | `RHO_A` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ 空气 |
| 336 | `backend/modules/green_shipping_iso19030/ship_model.py:38` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s 海水运动粘性（15 ℃） |
| 337 | `backend/modules/green_water/core.py:38` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 338 | `backend/modules/heading_keeping_sgisc/hull_wave_gm.py:62` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 |
| 339 | `backend/modules/heel_selfprop/aero.py:29` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³，15 ℃ 海平面 |
| 340 | `backend/modules/heel_selfprop/hull.py:39` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 341 | `backend/modules/heel_selfprop/hull.py:40` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 342 | `backend/modules/heel_selfprop/lateral.py:36` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 343 | `backend/modules/heel_selfprop/propeller.py:47` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 344 | `backend/modules/hull_form_optimization/geometry.py:57` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 345 | `backend/modules/hull_form_optimization/geometry.py:58` | `NU_SEAWATER` | 1.1883e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.0% | SI | m^2/s, ITTC-15 海水 |
| 346 | `backend/modules/hydrodynamics/shallow_water.py:48` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水 |
| 347 | `backend/modules/hydroelastic/slamming.py:43` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 348 | `backend/modules/hydroelastic/springing.py:51` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 349 | `backend/modules/ice_benchmark/models.py:61` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 350 | `backend/modules/ice_benchmark/models.py:62` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 351 | `backend/modules/ice_benchmark/models.py:64` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s at ~4 degC |
| 352 | `backend/modules/ice_bottoming/contact.py:231` | `nu_ice` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 353 | `backend/modules/ice_bottoming/contact.py:233` | `nu_steel` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 354 | `backend/modules/ice_bottoming/contact.py:251` | `mu_slow` | 0.1 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 355 | `backend/modules/ice_bottoming/contact.py:252` | `mu_fast` | 0.03 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 356 | `backend/modules/ice_bottoming/contact.py:253` | `mu_v_ref` | 0.75 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 357 | `backend/modules/ice_capability/capability.py:69` | `mu_scale` | 1 | review | `-` | - | - | dimensionless/derived coefficient | μ 敏感性乘子（判据 a-3 用） |
| 358 | `backend/modules/ice_closed_loop/dem.py:52` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | sigma_f (Pa) |
| 359 | `backend/modules/ice_closed_loop/dem.py:53` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | rho_i (kg/m3) |
| 360 | `backend/modules/ice_closed_loop/dem.py:54` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | rho_w (kg/m3) |
| 361 | `backend/modules/ice_closed_loop/dem.py:80` | `mu_ice_ice` | 0.3 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) |  |
| 362 | `backend/modules/ice_closed_loop/dem.py:81` | `mu_wall` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 363 | `backend/modules/ice_closed_loop/dem.py:86` | `nu_ice` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 364 | `backend/modules/ice_closed_loop/mmg_ice.py:34` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 365 | `backend/modules/ice_closed_loop/mmg_ice.py:35` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 366 | `backend/modules/ice_convoy/convoy.py:63` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 367 | `backend/modules/ice_crossimpl/lindqvist1989.py:94` | `nu_ice` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 368 | `backend/modules/ice_crossimpl/lindqvist1989.py:147` | `rho_ice` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 [kg/m^3] |
| 369 | `backend/modules/ice_crossimpl/lindqvist1989.py:148` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 [kg/m^3] |
| 370 | `backend/modules/ice_da/model_scale.py:50` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | kinematic viscosity of water [m^2/s] at ~15°C |
| 371 | `backend/modules/ice_da/model_scale.py:51` | `RHO_FW` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | fresh water (model basin) [kg/m^3] |
| 372 | `backend/modules/ice_da/model_scale.py:52` | `RHO_SW` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | sea water (full scale) [kg/m^3] |
| 373 | `backend/modules/ice_da/model_scale.py:65` | `ice_density_kgm3` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 374 | `backend/modules/ice_da/resistance.py:78` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 375 | `backend/modules/ice_da/resistance.py:79` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 376 | `backend/modules/ice_da/resistance.py:131` | `nu_kin` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 377 | `backend/modules/ice_damage_topology/flooding.py:59` | `MU_HOLD` | 0.6 | review | `-` | - | - | viscosity of an unlabelled substance | 货舱渗透率（一般散货/谷物；空舱 0.95 作敏感性端点） |
| 378 | `backend/modules/ice_damage_topology/flooding.py:60` | `MU_HOLD_EMPTY` | 0.95 | review | `-` | - | - | viscosity of an unlabelled substance | 空货舱（压载航行） |
| 379 | `backend/modules/ice_damage_topology/flooding.py:61` | `MU_TANK` | 0.95 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) | 空的双壳边舱（压载舱） |
| 380 | `backend/modules/ice_damage_topology/hull_lines.py:25` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水 |
| 381 | `backend/modules/ice_drift_roughness/model.py:34` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³, seawater density |
| 382 | `backend/modules/ice_drift_roughness/model.py:35` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m³, ice density |
| 383 | `backend/modules/ice_drift_roughness/model.py:36` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s, kinematic viscosity of seawater at 15°C |
| 384 | `backend/modules/ice_dual_arb/dem.py:42` | `MU_RUBBLE` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 碎冰-碎冰/水摩擦系数（rubble friction） |
| 385 | `backend/modules/ice_dual_arb/models.py:35` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 海冰密度 [kg/m³] |
| 386 | `backend/modules/ice_dual_arb/models.py:36` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 387 | `backend/modules/ice_dual_arb/models.py:38` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 [-] |
| 388 | `backend/modules/ice_dual_arb/models.py:39` | `MU_ICE_STEEL` | 0.15 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-钢摩擦系数 [-] |
| 389 | `backend/modules/ice_dual_arb/models.py:101` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | 弯曲强度 σf [Pa]（一年冰 300-700 kPa） |
| 390 | `backend/modules/ice_fatigue/assessment.py:63` | `ultimate_strength` | 500 | review | `-` | - | - | unclassified | MPa |
| 391 | `backend/modules/ice_fatigue/fatigue_life.py:54` | `ultimate_strength` | 500 | review | `-` | - | - | unclassified |  |
| 392 | `backend/modules/ice_fatigue/ice_load_psd.py:159` | `density` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | Ice density [kg/m^3] |
| 393 | `backend/modules/ice_fatigue/ice_load_psd.py:160` | `crushing_strength` | 5 | review | `-` | - | - | unclassified | Uniaxial crushing strength [MPa] |
| 394 | `backend/modules/ice_fatigue/ice_load_psd.py:366` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 395 | `backend/modules/ice_fatigue/miner_damage.py:30` | `ultimate_strength` | 500 | review | `-` | - | - | unclassified |  |
| 396 | `backend/modules/ice_fatigue/rainflow.py:247` | `ultimate_strength` | 500 | review | `-` | - | - | unclassified |  |
| 397 | `backend/modules/ice_fatigue/sn_arctic.py:90` | `ultimate_strength` | 500 | review | `-` | - | - | unclassified | MPa |
| 398 | `backend/modules/ice_fatigue/sn_arctic.py:91` | `yield_strength` | 355 | review | `-` | - | - | yield stress, steel grade not identified in the code | MPa |
| 399 | `backend/modules/ice_fatigue_spectrum/assessment.py:50` | `STEEL_DENSITY_KG_M3` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 400 | `backend/modules/ice_fatigue_spectrum/beam_dynamics.py:111` | `young_modulus_pa` | 2.069e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 401 | `backend/modules/ice_fatigue_spectrum/ice_force.py:44` | `DEFAULT_FLEXURAL_STRENGTH_PA` | 500000 | review | `-` | - | - | unclassified |  |
| 402 | `backend/modules/ice_flooding/topology.py:149` | `mu_wing` | 0.95 | review | `-` | - | - | viscosity of an unlabelled substance | permeability of the double-side void |
| 403 | `backend/modules/ice_flooding/topology.py:150` | `mu_hold` | 0.7 | review | `-` | - | - | viscosity of an unlabelled substance | permeability of a cargo hold (cargo/structure |
| 404 | `backend/modules/ice_fsi/ice_force.py:58` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 405 | `backend/modules/ice_fsi/ice_force.py:74` | `nu_ice` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 406 | `backend/modules/ice_fsi/icing.py:64` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水密度 |
| 407 | `backend/modules/ice_fsi/structure_dyn.py:108` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 408 | `backend/modules/ice_fullscale/fullscale_reference.py:115` | `MU_REF` | 0.55 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 409 | `backend/modules/ice_fullscale/ice_physics.py:46` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 410 | `backend/modules/ice_fullscale/ice_physics.py:47` | `RHO_SNOW` | 300 | review | `-` | - | - | density of an unlabelled substance |  |
| 411 | `backend/modules/ice_fullscale/ice_physics.py:48` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 412 | `backend/modules/ice_fullscale/ice_physics.py:50` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 |
| 413 | `backend/modules/ice_fullscale/ice_physics.py:52` | `MU_V_REF` | 0.6 | review | `-` | - | - | viscosity of an unlabelled substance | 摩擦速度软化参考速度 [m/s] |
| 414 | `backend/modules/ice_fullscale/ice_physics.py:53` | `MU_MIN_RATIO` | 0.55 | review | `-` | - | - | dimensionless/derived coefficient | 高速极限摩擦系数 / μ0 |
| 415 | `backend/modules/ice_fullscale/ice_physics.py:141` | `MU_V_REF_SAMPLE` | 1 | review | `-` | - | - | viscosity of an unlabelled substance | 取摩擦系数的参考滑移速度 [m/s] |
| 416 | `backend/modules/ice_fullscale/lindqvist_keinonen.py:61` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | sea water [kg/m^3] |
| 417 | `backend/modules/ice_fullscale/lindqvist_keinonen.py:62` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | sea ice [kg/m^3] |
| 418 | `backend/modules/ice_fullscale/lindqvist_keinonen.py:64` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | ice Poisson's ratio |
| 419 | `backend/modules/ice_fullscale/vessels.py:17` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水 |
| 420 | `backend/modules/ice_interaction_anchor/friction.py:44` | `MU_REF` | 0.045 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 421 | `backend/modules/ice_interaction_anchor/params.py:22` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m^3] |
| 422 | `backend/modules/ice_interaction_anchor/params.py:23` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 海冰密度 [kg/m^3] |
| 423 | `backend/modules/ice_interaction_anchor/params.py:24` | `RHO_SNOW` | 300 | review | `-` | - | - | density of an unlabelled substance | 积雪密度 [kg/m^3] |
| 424 | `backend/modules/ice_interaction_anchor/params.py:26` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 |
| 425 | `backend/modules/ice_interaction_validation/brash.py:41` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 426 | `backend/modules/ice_interaction_validation/ledger.py:43` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 427 | `backend/modules/ice_interaction_validation/plate_bending.py:64` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 泊松比 |
| 428 | `backend/modules/ice_interaction_validation/plate_bending.py:65` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 429 | `backend/modules/ice_lindqvist/common.py:25` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 430 | `backend/modules/ice_lindqvist/common.py:26` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 海冰密度 [kg/m³]（一年冰 890~920） |
| 431 | `backend/modules/ice_lindqvist/common.py:28` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 [-] |
| 432 | `backend/modules/ice_lindqvist/common.py:30` | `MU_ICE_STEEL` | 0.1 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-钢摩擦系数（Lindqvist 观测 0.05~0.15，取 0.10） |
| 433 | `backend/modules/ice_lindqvist/common.py:31` | `RHO_SNOW` | 300 | review | `-` | - | - | density of an unlabelled substance | 积雪密度 [kg/m³]（干雪 250~350） |
| 434 | `backend/modules/ice_lindqvist_1983/lindqvist_1983.py:83` | `rho_ice` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 435 | `backend/modules/ice_lindqvist_1983/lindqvist_1983.py:84` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 436 | `backend/modules/ice_load_pod/dem.py:68` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 437 | `backend/modules/ice_load_pod/dem.py:69` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 438 | `backend/modules/ice_load_pod/dem.py:101` | `strength_mean` | 2e+06 | review | `-` | - | - | unclassified | Pa    块元破碎强度均值 |
| 439 | `backend/modules/ice_load_pod/dem.py:102` | `strength_modulus` | 3 | review | `-` | - | - | unclassified | [-]   Weibull 模数 m |
| 440 | `backend/modules/ice_load_pod/dem.py:111` | `strength_corr_factor` | 0.6 | review | `-` | - | - | dimensionless/derived coefficient | [-]   强度随机场的空间相关长度 = factor · L_share |
| 441 | `backend/modules/ice_load_pod/dem.py:112` | `strength_corr_weight` | 0.7 | review | `-` | - | - | unclassified | [-]   相关分量在强度随机性中的权重（0=纯白噪声） |
| 442 | `backend/modules/ice_maneuvering/hydro.py:33` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 443 | `backend/modules/ice_maneuvering/ice_force.py:49` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 444 | `backend/modules/ice_maneuvering/ice_force.py:51` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 445 | `backend/modules/ice_maneuvering/ice_force.py:80` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | 弯曲强度 σ_f [Pa] |
| 446 | `backend/modules/ice_maneuvering/ship.py:22` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m3] |
| 447 | `backend/modules/ice_maneuvering/ship.py:23` | `RHO_A` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 [kg/m3] |
| 448 | `backend/modules/ice_maneuvering_capability/capability_map.py:56` | `ice_flexural_strength` | 500000 | review | `-` | - | - | unclassified | 冰弯曲强度 (Pa) |
| 449 | `backend/modules/ice_maneuvering_capability/capability_map.py:57` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 450 | `backend/modules/ice_maneuvering_capability/capability_map.py:130` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 451 | `backend/modules/ice_maneuvering_capability/capability_map.py:131` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 kg/m³ |
| 452 | `backend/modules/ice_moving_load/plate.py:57` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | sea water |
| 453 | `backend/modules/ice_moving_load/plate.py:58` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | columnar sea ice |
| 454 | `backend/modules/ice_mt_uikku/lindqvist3.py:55` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 455 | `backend/modules/ice_mt_uikku/lindqvist3.py:56` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 456 | `backend/modules/ice_mt_uikku/lindqvist3.py:58` | `NU_ICE` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 457 | `backend/modules/ice_mt_uikku/uq.py:41` | `mu_low` | 0.1 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 458 | `backend/modules/ice_mt_uikku/uq.py:42` | `mu_high` | 0.2 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 459 | `backend/modules/ice_oil_spill/constants.py:39` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 460 | `backend/modules/ice_oil_spill/constants.py:40` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 461 | `backend/modules/ice_oil_spill/constants.py:41` | `NU_W` | 1.8e-06 | review | `-` | - | - | viscosity of an unlabelled substance | m²/s |
| 462 | `backend/modules/ice_oil_spill/drift.py:67` | `RHO_AIR` | 1.3 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 463 | `backend/modules/ice_oil_spill/drift.py:93` | `eddy_viscosity` | 0.1 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) | m²/s, ice-ocean BBL (D_E ~ 38 m at 75 deg N) |
| 464 | `backend/modules/ice_oil_spill/simulation.py:70` | `oil_density` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 465 | `backend/modules/ice_oil_spill/simulation.py:71` | `oil_viscosity_pas` | 0.005 | review | `-` | - | - | viscosity of an unlabelled substance | Pa·s |
| 466 | `backend/modules/ice_oil_spill/tests/test_ice_oil_spill.py:143` | `rho_oil` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) |  |
| 467 | `backend/modules/ice_oil_spill/tests/test_ice_oil_spill.py:144` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 468 | `backend/modules/ice_oil_spill/tests/test_ice_oil_spill.py:145` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 469 | `backend/modules/ice_oil_spill/tests/test_ice_oil_spill.py:157` | `rho_oil` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) |  |
| 470 | `backend/modules/ice_oil_spill/tests/test_ice_oil_spill.py:158` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 471 | `backend/modules/ice_oil_spill/tests/test_ice_oil_spill.py:159` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 472 | `backend/modules/ice_operability/vibration.py:64` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 473 | `backend/modules/ice_operability_domain/domain.py:96` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | 弯曲强度 (Pa) |
| 474 | `backend/modules/ice_operability_domain/domain.py:97` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 (kg/m³) |
| 475 | `backend/modules/ice_operability_domain/domain.py:98` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 476 | `backend/modules/ice_operability_domain/domain.py:272` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 kg/m³ |
| 477 | `backend/modules/ice_parametric_roll/envelope.py:32` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 478 | `backend/modules/ice_plate_limit/panel.py:42` | `DEFAULT_YIELD_PA` | 3.55e+08 | review | `-` | - | - | unclassified |  |
| 479 | `backend/modules/ice_pod/loadfield.py:65` | `crush_strength_pa` | 1.5e+06 | review | `-` | - | - | unclassified | 名义压溃强度 p_c [Pa] |
| 480 | `backend/modules/ice_pod/loadfield.py:70` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 481 | `backend/modules/ice_pod/loadfield.py:74` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 海冰密度 [kg/m^3]（碎冰质量 = rho_ice pi R^2 h） |
| 482 | `backend/modules/ice_pod_ekf/cases.py:26` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 483 | `backend/modules/ice_pod_ekf/dem.py:58` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 484 | `backend/modules/ice_pod_ekf/dem.py:59` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 485 | `backend/modules/ice_pod_ekf/dem.py:68` | `nu_ice` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 486 | `backend/modules/ice_property_uq/properties.py:270` | `MU_MEAN` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 487 | `backend/modules/ice_property_uq/properties.py:271` | `MU_COV` | 0.3333 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 488 | `backend/modules/ice_property_uq/resistance.py:72` | `NU_ICE` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | Poisson ratio |
| 489 | `backend/modules/ice_property_uq/resistance.py:73` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m3, seawater |
| 490 | `backend/modules/ice_property_uq/resistance.py:74` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m3, ice |
| 491 | `backend/modules/ice_radiation_dynamics/floe_geometries.py:35` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 492 | `backend/modules/ice_radiation_dynamics/floe_geometries.py:120` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 493 | `backend/modules/ice_radiation_dynamics/floe_geometries.py:253` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 494 | `backend/modules/ice_radiation_dynamics/floe_geometries.py:398` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 495 | `backend/modules/ice_radiation_dynamics/tests/test_radiation_dynamics.py:56` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 496 | `backend/modules/ice_radiation_dynamics/tests/test_radiation_dynamics.py:82` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 497 | `backend/modules/ice_rao/ice_rao.py:129` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 498 | `backend/modules/ice_rao/ice_rao.py:227` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 499 | `backend/modules/ice_rao/tests/test_ice_rao.py:188` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 500 | `backend/modules/ice_rao/tests/test_ice_rao.py:267` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 501 | `backend/modules/ice_rao/tests/test_ice_rao.py:288` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 502 | `backend/modules/ice_resistance_chain/anchors.py:38` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 503 | `backend/modules/ice_resistance_chain/anchors.py:39` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 海冰密度 kg/m³ |
| 504 | `backend/modules/ice_resistance_chain/anchors.py:41` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 泊松比 |
| 505 | `backend/modules/ice_ridge/resistance.py:116` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | ice density [kg/m^3] |
| 506 | `backend/modules/ice_ridge/resistance.py:117` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | water density [kg/m^3] |
| 507 | `backend/modules/ice_ridge/resistance.py:122` | `mu_steel` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance | ice-steel friction |
| 508 | `backend/modules/ice_ridge/resistance.py:123` | `mu_rubble` | 0.25 | review | `-` | - | - | viscosity of an unlabelled substance | rubble-ice / steel friction |
| 509 | `backend/modules/ice_ridge/stability.py:139` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 510 | `backend/modules/ice_ridge_ramming/ice_force.py:60` | `strength_scale` | 1 | review | `-` | - | - | dimensionless/derived coefficient | 多次未破脊后的强度折减（1.0 = 完整冰脊） |
| 511 | `backend/modules/ice_ridge_riska/geometry.py:20` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 kg/m³ |
| 512 | `backend/modules/ice_ridge_riska/geometry.py:21` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 513 | `backend/modules/ice_ridge_riska/geometry.py:29` | `MU_HULL` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance | 冰-钢摩擦系数（0.10–0.20） |
| 514 | `backend/modules/ice_ridge_thermo/thermal.py:24` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 冰密度 kg/m3 |
| 515 | `backend/modules/ice_ridge_thermo/thermal.py:27` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m3 |
| 516 | `backend/modules/ice_roughness/drift.py:38` | `RHO_AIR` | 1.3 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 517 | `backend/modules/ice_roughness/drift.py:39` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 518 | `backend/modules/ice_roughness/drift.py:40` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 519 | `backend/modules/ice_roughness/drift.py:43` | `NU_WATER` | 1.8e-06 | review | `-` | - | - | viscosity of an unlabelled substance | m^2/s, sea water at ~0 C |
| 520 | `backend/modules/ice_rudder_jamming/steering_gear.py:111` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 521 | `backend/modules/ice_scale_calib/bayes.py:43` | `NU_T` | 6 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) |  |
| 522 | `backend/modules/ice_scale_calib/dataset.py:57` | `NU_T` | 6 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) | 重尾自由度（全尺度偶发冰脊/碎冰冲击） |
| 523 | `backend/modules/ice_scale_calib/scaling.py:45` | `ICE_FLEXURAL_STRENGTH_PA` | 500000 | review | `-` | - | - | unclassified |  |
| 524 | `backend/modules/ice_scaling/formulas.py:84` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 525 | `backend/modules/ice_stability_cargo/stability_coupled.py:128` | `ice_density_kg_m3` | 890 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | 冰密度 [kg/m³] |
| 526 | `backend/modules/ice_stability_coupling/acceptance.py:39` | `ICE_DENSITY_KG_M3` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 积冰密度（IS Code 取值） |
| 527 | `backend/modules/ice_stability_coupling/ship.py:43` | `water_density_kg_m3` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 528 | `backend/modules/ice_thermo/column.py:62` | `snow_density0` | 300 | review | `-` | - | - | density of an unlabelled substance | kg/m^3 |
| 529 | `backend/modules/ice_thermo/constants.py:24` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg m^-3 |
| 530 | `backend/modules/ice_thermo/constants.py:25` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg m^-3 (sea water) |
| 531 | `backend/modules/ice_thermo/constants.py:26` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg m^-3 (melt water) |
| 532 | `backend/modules/ice_thermo/constants.py:27` | `RHO_SNOW_FRESH` | 250 | conflict | `fresh_water/density` | 999.1 kg/m^3 | 75.0% | SI | kg m^-3, density of newly fallen snow |
| 533 | `backend/modules/ice_thermo/constants.py:28` | `RHO_SNOW_MAX` | 450 | review | `-` | - | - | density of an unlabelled substance | kg m^-3, densification ceiling |
| 534 | `backend/modules/ice_thermo/forcing.py:135` | `snow_density` | 300 | review | `-` | - | - | density of an unlabelled substance | kg/m^3 |
| 535 | `backend/modules/ice_twin_maturity/ice_field.py:66` | `SIGMA_YOUNG` | 1.05e+06 | review | `-` | - | - | yield stress, steel grade not identified in the code | Pa  compressive strength, young/thin ice |
| 536 | `backend/modules/ice_twin_maturity/reference.py:75` | `strength_scatter` | 0.15 | review | `-` | - | - | unclassified |  |
| 537 | `backend/modules/ice_twin_pod/floe_dem.py:47` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 538 | `backend/modules/ice_twin_pod/floe_dem.py:48` | `RHO_I` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 539 | `backend/modules/ice_uikku/formula_forms.py:33` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 540 | `backend/modules/ice_uikku/formula_forms.py:34` | `RHO_I` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 541 | `backend/modules/ice_uikku/formula_forms.py:35` | `NU_ICE` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 542 | `backend/modules/ice_uikku/formula_forms.py:36` | `MU_ICE_STEEL` | 0.15 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-钢摩擦系数工程值（0.10-0.15 淡水/波罗的海冰） |
| 543 | `backend/modules/ice_velocity_exponent/flume_dem.py:59` | `MU_RUBBLE` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 碎冰-碎冰/水摩擦（与 dem.py 一致） |
| 544 | `backend/modules/ice_velocity_exponent/flume_dem.py:82` | `mu_hull` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance | 冰-钢（m9406 models.MU_ICE_STEEL） |
| 545 | `backend/modules/ice_velocity_exponent/flume_dem.py:83` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified |  |
| 546 | `backend/modules/ice_vibration/model.py:118` | `crushing_strength` | 1e+06 | review | `-` | - | - | unclassified |  |
| 547 | `backend/modules/ice_wave_coupling/coupled.py:55` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 548 | `backend/modules/ice_wave_coupling/hydro.py:47` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 549 | `backend/modules/ice_wave_coupling/ice_field.py:73` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | 弯曲强度 σ_f 均值 [Pa] |
| 550 | `backend/modules/ice_wave_coupling/ice_field.py:74` | `strength_cov` | 0.2 | review | `-` | - | - | unclassified | 弯曲强度变异系数（冰材性统计，DECLARED） |
| 551 | `backend/modules/ice_wave_coupling/ice_field.py:75` | `ice_density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | ρ_i [kg/m³] |
| 552 | `backend/modules/ice_wave_coupling/ice_field.py:76` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | ρ_w [kg/m³] |
| 553 | `backend/modules/ice_whipping/girder.py:46` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 554 | `backend/modules/keel_gouging/contact_spectrum.py:171` | `crushing_strength` | 2.5e+06 | review | `-` | - | - | unclassified | sigma_c [Pa] 受约束压碎强度 |
| 555 | `backend/modules/keel_gouging/grounding_resistance.py:79` | `crushing_strength` | 2.5e+06 | review | `-` | - | - | unclassified | sigma_c [Pa] |
| 556 | `backend/modules/keel_gouging/grounding_resistance.py:81` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 557 | `backend/modules/keel_gouging/grounding_resistance.py:82` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 558 | `backend/modules/lng_thermal/bog_ledger.py:34` | `RHO_LNG` | 425 | review | `-` | - | - | density of an unlabelled substance | [kg/m³] 液相密度 |
| 559 | `backend/modules/lng_thermal/chain.py:39` | `delta_rho` | 50 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 560 | `backend/modules/loading_stability/geometry.py:70` | `RHO_SEA` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m^3 海水密度 (与平台其它模块口径一致) |
| 561 | `backend/modules/loading_stability/strength.py:66` | `YIELD_STEEL_MPA` | 235 | review | `-` | - | - | unclassified |  |
| 562 | `backend/modules/loading_stability/tanks.py:40` | `RHO_SEA` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 563 | `backend/modules/lsa_launch/airborne.py:26` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 564 | `backend/modules/lsa_launch/criteria.py:41` | `DAVIT_STRENGTH_FACTOR` | 1.1 | review | `-` | - | - | dimensionless/derived coefficient | 吊架强度因数: SWL ≥ 1.1 × 满载艇重 |
| 565 | `backend/modules/lsa_launch/entry.py:42` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 566 | `backend/modules/lsa_launch/geometry.py:31` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m3 海水 |
| 567 | `backend/modules/lsa_launch/wedge.py:57` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 568 | `backend/modules/manoeuvring_mmg.py:21` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 569 | `backend/modules/mooring/catenary.py:56` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] |
| 570 | `backend/modules/mooring/catenary.py:57` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | [kg/m³] |
| 571 | `backend/modules/mooring/environment.py:42` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 572 | `backend/modules/open_water_fatigue/geometry_gate.py:19` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水 |
| 573 | `backend/modules/open_water_fatigue/stress_rao.py:25` | `MU_ADDED` | 1 | review | `-` | - | - | viscosity of an unlabelled substance | 附加质量比（声明） |
| 574 | `backend/modules/polar_class/ur1_rules.py:26` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水 |
| 575 | `backend/modules/polar_class/ur1_rules.py:27` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3 冰 |
| 576 | `backend/modules/polar_class/ur1_rules.py:34` | `NU_ICE_DEFAULT` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | -   泊松比 |
| 577 | `backend/modules/polar_class/ur1_rules.py:46` | `MU_ICE` | 0.15 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | -   冰-钢摩擦 |
| 578 | `backend/modules/polaris/hull_load.py:48` | `ICE_COMPRESSIVE_STRENGTH_DEFAULT` | 3 | review | `-` | - | - | unclassified | MPa，中等强度一年冰 |
| 579 | `backend/modules/polaris/hull_load.py:50` | `ICE_FLEXURAL_STRENGTH_DEFAULT` | 0.7 | review | `-` | - | - | unclassified | MPa |
| 580 | `backend/modules/polaris/hull_load.py:54` | `SEA_WATER_DENSITY` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 581 | `backend/modules/polaris/hull_load.py:56` | `ICE_DENSITY` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 582 | `backend/modules/potential_flow/sphere_radiation.py:85` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 583 | `backend/modules/propeller_bemt.py:58` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 584 | `backend/modules/propeller_bemt.py:59` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s，海水运动粘度（15 ℃） |
| 585 | `backend/modules/propeller_emergence/campaign.py:44` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 586 | `backend/modules/propeller_emergence/shaft.py:64` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 587 | `backend/modules/propeller_hull_interaction/mmg_propeller_closure.py:54` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 588 | `backend/modules/propeller_ice_blockage/degradation.py:66` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 589 | `backend/modules/propeller_ice_blockage/pipeline.py:45` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 590 | `backend/modules/propeller_ice_impact/contact.py:70` | `density_kgm3` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 591 | `backend/modules/propeller_ice_ventilation/bubble.py:38` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 592 | `backend/modules/propeller_ice_ventilation/bubble_dynamics.py:52` | `density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 593 | `backend/modules/propeller_ice_ventilation/bubble_dynamics.py:53` | `viscosity` | 0.00107 | review | `-` | - | - | viscosity of an unlabelled substance | Pa·s (dynamic viscosity at 15°C) |
| 594 | `backend/modules/propeller_ice_ventilation/config.py:19` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m3  海水密度 |
| 595 | `backend/modules/propeller_ice_ventilation/config.py:20` | `NU_SEA_WATER` | 1.35e-06 | conflict | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 13.6% | SI | m2/s   海水运动粘度 (10 degC, S=35) |
| 596 | `backend/modules/propeller_ice_ventilation/config.py:41` | `ice_strength_ref` | 5.8e+06 | review | `-` | - | - | unclassified | Pa  reference 压溃压力 @ A_ref（见 contact.py） |
| 597 | `backend/modules/propeller_ice_ventilation/config.py:44` | `ice_strength_cv` | 0.12 | review | `-` | - | - | unclassified | -      冰强度非均质变异系数（seeded，scenario input） |
| 598 | `backend/modules/propeller_ice_ventilation/contact.py:51` | `strength_cv` | 0.12 | review | `-` | - | - | unclassified |  |
| 599 | `backend/modules/propeller_ice_ventilation/contact_mechanics.py:43` | `density` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 600 | `backend/modules/propeller_ice_ventilation/contact_mechanics.py:45` | `poisson_ratio` | 0.33 | review | `-` | - | - | dimensionless/derived coefficient |  |
| 601 | `backend/modules/propeller_ice_ventilation/contact_mechanics.py:46` | `flexural_strength` | 1e+06 | review | `-` | - | - | unclassified | Pa (1 MPa) |
| 602 | `backend/modules/propeller_ice_ventilation/contact_mechanics.py:47` | `crushing_strength` | 5e+06 | review | `-` | - | - | unclassified | Pa (5 MPa) |
| 603 | `backend/modules/propeller_rudder_interaction.py:65` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 604 | `backend/modules/propeller_selfprop.py:73` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³，海水密度（15 ℃） |
| 605 | `backend/modules/propeller_ventilation.py:61` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³，20°C 标准大气（仅用于掺气质量分数换算） |
| 606 | `backend/modules/propulsion/cavitation.py:80` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 607 | `backend/modules/propulsion/reference_provenance.py:30` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 608 | `backend/modules/propulsion_chain.py:73` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 609 | `backend/modules/propulsion_chain.py:74` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 610 | `backend/modules/propulsion_ice/shaft_lateral.py:84` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 [kg/m³] |
| 611 | `backend/modules/sea_ice_thermo/constants.py:15` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m3 |
| 612 | `backend/modules/sea_ice_thermo/constants.py:18` | `rho_snow` | 330 | review | `-` | - | - | density of an unlabelled substance | kg/m3    (Arctic snow, Sturm et al. 2002) |
| 613 | `backend/modules/sea_ice_thermo/constants.py:25` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m3 |
| 614 | `backend/modules/sea_ice_thermo/constants.py:28` | `rho_a` | 1.3 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m3 |
| 615 | `backend/modules/seakeeping/wave_drift.py:46` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 616 | `backend/modules/seakeeping_rao/geometry.py:37` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 617 | `backend/modules/seakeeping_rao/ikeda.py:52` | `NU_KINEMATIC` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 618 | `backend/modules/self_propulsion.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 619 | `backend/modules/sgisc_l2/criteria.py:51` | `NU_KIN` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 海水运动粘性系数 [m²/s] @15°C |
| 620 | `backend/modules/sgisc_l2/hull.py:57` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 621 | `backend/modules/sgisc_l2/seastate.py:44` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 622 | `backend/modules/shallow_maneuvering/squat.py:65` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 623 | `backend/modules/shallow_water/hull.py:27` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 624 | `backend/modules/shallow_water/tuck_barrass.py:106` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 625 | `backend/modules/shallow_water_two_ship/analytical_2d.py:47` | `density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 密度 ρ (kg/m³) |
| 626 | `backend/modules/sr_rudder_opt/objective.py:70` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 627 | `backend/modules/structural/slamming.py:28` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m^3] |
| 628 | `backend/modules/underwater_acoustics/analytical.py:13` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 629 | `backend/modules/waterjet_cfd/calibrate.py:124` | `rho_j` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 630 | `backend/modules/waterjet_cfd/geometry.py:153` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 631 | `backend/modules/waterjet_cfd/geometry.py:154` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s (15 degC 海水) |
| 632 | `backend/modules/weather_routing/core.py:49` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水密度 |
| 633 | `backend/modules/weather_routing/core.py:50` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 空气密度 |
| 634 | `backend/modules/weather_routing/ship_response.py:50` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 [kg/m³] |
| 635 | `backend/modules/weather_routing/ship_response.py:51` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 636 | `backend/modules/weather_routing/ship_response.py:53` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 海水运动粘度 [m²/s] |
| 637 | `backend/modules/weather_routing/weather_models.py:158` | `mu_wind` | 10 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 638 | `backend/modules/weather_routing/weather_models.py:160` | `mu_wave` | 2 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 639 | `backend/modules/weather_routing/weather_models.py:162` | `mu_current` | 0.5 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 640 | `backend/modules/wind_assist/magnus.py:77` | `AIR_DENSITY` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ @15°C, 101.325 kPa（ISA 海平面） |
| 641 | `backend/modules/wind_assist/magnus.py:78` | `AIR_KINEMATIC_VISCOSITY` | 1.5e-05 | migrate | `air/viscosity_kinematic` | 1.46065e-05 m^2/s | 2.7% | SI | m²/s @15°C |
| 642 | `backend/modules/wind_assist/propeller.py:27` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 643 | `backend/modules/wind_assist/rotor_aero.py:70` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3, ISA 海平面 15℃ |
| 644 | `backend/modules/wind_assist/rotor_aero.py:71` | `NU_AIR` | 1.48e-05 | migrate | `air/viscosity_kinematic` | 1.46065e-05 m^2/s | 1.3% | SI | m^2/s, 15℃ 空气运动粘度 |
| 645 | `backend/modules/wind_assist/ship_balance.py:43` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 646 | `backend/modules/wind_heel_weather/criterion.py:56` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3, 15 degC, 1013 hPa |
| 647 | `backend/modules/wind_heel_weather/criterion.py:64` | `RHO_SEA` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m^3 -> mass from displacement volume |
| 648 | `backend/modules/wind_load_ocimf.py:41` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ (15°C 标准大气) |
| 649 | `backend/modules/wind_loads/blendermann.py:88` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3, 15 degC standard atmosphere |
| 650 | `backend/moonpool/moonpool_solver.py:74` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 651 | `backend/physics/acv/closed_chain.py:49` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 15°C 海水运动粘性 (m²/s) |
| 652 | `backend/physics/acv/leakage.py:40` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 标准空气密度 (kg/m³) |
| 653 | `backend/physics/acv/wave_resistance.py:65` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 (kg/m³) |
| 654 | `backend/physics/added_resistance_spectral.py:106` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 655 | `backend/physics/added_resistance_waves.py:65` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 656 | `backend/physics/air_lubrication/artifacts.py:42` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 657 | `backend/physics/air_lubrication/dispersion.py:69` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 kg/m^3 |
| 658 | `backend/physics/air_lubrication/dispersion.py:70` | `rho_a` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 kg/m^3 |
| 659 | `backend/physics/air_lubrication/dispersion.py:73` | `nu_w` | 1.05e-06 | migrate | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 7.8% | SI | 水运动粘度 m^2/s（粘性阻尼判据用） |
| 660 | `backend/physics/air_lubrication/dispersion.py:74` | `nu_a` | 1.48e-05 | review | `-` | - | - | viscosity of an unlabelled substance | 空气运动粘度 m^2/s |
| 661 | `backend/physics/air_lubrication/interface_eigen.py:45` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水密度 |
| 662 | `backend/physics/air_lubrication/interface_eigen.py:46` | `RHO_A` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 空气密度 |
| 663 | `backend/physics/air_lubrication/interface_eigen.py:49` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s 海水运动粘度 |
| 664 | `backend/physics/air_lubrication/net_power.py:48` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 665 | `backend/physics/air_resistance.py:28` | `RHO_AIR_DEFAULT` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 标准大气密度 [kg/m³] (ISA, 15°C, 101325 Pa) |
| 666 | `backend/physics/air_resistance.py:29` | `RHO_AIR_COLD` | 1.292 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 冷空气密度 [kg/m³] (0°C) |
| 667 | `backend/physics/air_resistance.py:30` | `RHO_AIR_TROPICAL` | 1.112 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) | 热带空气密度 [kg/m³] (35°C) |
| 668 | `backend/physics/anchor_equipment/assessment.py:45` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 669 | `backend/physics/anchor_equipment/catenary.py:47` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 670 | `backend/physics/anchor_equipment/catenary.py:48` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 671 | `backend/physics/anchor_equipment/equipment.py:39` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 672 | `backend/physics/anchor_equipment/loads.py:42` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 @ 15 °C |
| 673 | `backend/physics/anchor_equipment/loads.py:43` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 674 | `backend/physics/ballast_exchange/hydraulics.py:39` | `RHO_SW` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m^3 |
| 675 | `backend/physics/ballast_exchange/pump.py:43` | `RHO_SEAWATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m3 |
| 676 | `backend/physics/ballast_exchange/tank.py:37` | `RHO_SW` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 海水密度 t/m^3 |
| 677 | `backend/physics/ballast_exchange/tanks.py:37` | `RHO_BALLAST` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 海水压载 t/m3（BWM 置换对象即海水） |
| 678 | `backend/physics/ballast_exchange.py:46` | `rho_liquid` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 液体密度 [t/m³] |
| 679 | `backend/physics/ballast_exchange.py:71` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 680 | `backend/physics/ballast_stability_td/geometry.py:150` | `rho` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 液体密度 [t/m³] |
| 681 | `backend/physics/ballast_stability_td/ship.py:27` | `RHO_W` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 海水密度 t/m³ |
| 682 | `backend/physics/bearing_dynamics/rotor_dynamics.py:14` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 683 | `backend/physics/biofouling/resistance.py:32` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 684 | `backend/physics/biofouling/resistance.py:33` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s |
| 685 | `backend/physics/biofouling_progression/resistance.py:50` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 686 | `backend/physics/biofouling_progression/resistance.py:51` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s，15 °C 海水运动粘性（与平台其它模块一致） |
| 687 | `backend/physics/brash_coupling/field.py:19` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³（冰区低温取 1025） |
| 688 | `backend/physics/brash_coupling/field.py:20` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 海冰密度 kg/m³ |
| 689 | `backend/physics/broken_ice_maneuvering/ice_field.py:61` | `MU_ICE_HULL` | 0.15 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-船摩擦系数 |
| 690 | `backend/physics/broken_ice_maneuvering/ship.py:21` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 691 | `backend/physics/broken_ice_maneuvering/ship.py:22` | `RHO_I` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 692 | `backend/physics/broken_ice_maneuvering/ship.py:23` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 693 | `backend/physics/bubble_dynamics.py:71` | `rho` | 998.2 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 液体密度 [kg/m3] |
| 694 | `backend/physics/cabin_noise/__init__.py:99` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 695 | `backend/physics/cabin_noise/__init__.py:101` | `RHO_C_REF` | 415 | review | `-` | - | - | density of an unlabelled substance | Rayl (标准参考值) |
| 696 | `backend/physics/cabin_noise/__init__.py:253` | `rho_s` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 697 | `backend/physics/cargo_securing/constants.py:24` | `RHO_SEAWATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 698 | `backend/physics/cargo_securing/loads.py:72` | `hatch_cover_material_yield` | 235 | review | `-` | - | - | unclassified | MPa (mild steel) |
| 699 | `backend/physics/cargo_securing/loads.py:255` | `corner_strength` | 864 | review | `-` | - | - | unclassified | kN |
| 700 | `backend/physics/cathodic_recycling/cp_model.py:53` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 701 | `backend/physics/cavitation_bucket.py:46` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 702 | `backend/physics/cavitation_noise.py:55` | `RHO_L` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 水密度 kg/m³ (20°C) |
| 703 | `backend/physics/cavitation_noise.py:56` | `RHO_G` | 1.2 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 kg/m³ (20°C, 1atm) |
| 704 | `backend/physics/composite_prop/added_mass.py:49` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³，海水（15 ℃） |
| 705 | `backend/physics/composite_prop/fsi.py:95` | `core_density` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³，夹芯填充密度 |
| 706 | `backend/physics/composite_prop/section.py:55` | `core_density` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³，夹芯填充（仅贡献质量/转动惯量） |
| 707 | `backend/physics/design_spiral/spiral.py:94` | `cargo_density` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 货物密度 (t/m³) |
| 708 | `backend/physics/design_spiral/volume_check.py:63` | `cargo_density` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 货物密度 (t/m³) |
| 709 | `backend/physics/dp_capability/environment.py:38` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | [kg/m^3] 15 degC 标准空气密度 |
| 710 | `backend/physics/dp_capability/environment.py:39` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 711 | `backend/physics/drag_interaction/baseline.py:26` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 712 | `backend/physics/drag_interaction/baseline.py:27` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 713 | `backend/physics/drag_interaction/baseline.py:28` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 714 | `backend/physics/drag_interaction/lbm.py:49` | `MU_RATIO` | 55 | review | `-` | - | - | dimensionless/derived coefficient |  |
| 715 | `backend/physics/drag_interaction/lbm.py:59` | `nu_liquid` | 0.2 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 716 | `backend/physics/dredging/anchors.py:22` | `RHO_WATER` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m^3, 清水（挖泥船多在淡水/微咸水域作业，取淡水为基准） |
| 717 | `backend/physics/dredging/anchors.py:23` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水（备用） |
| 718 | `backend/physics/dredging/anchors.py:24` | `RHO_SAND` | 2650 | review | `-` | - | - | density of an unlabelled substance | kg/m^3, 石英砂颗粒密度 |
| 719 | `backend/physics/dredging/anchors.py:25` | `NU_WATER` | 1.004e-06 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 11.8% | SI | m^2/s, 20 degC 清水运动粘度 |
| 720 | `backend/physics/dredging/anchors.py:26` | `MU_WATER` | 0.001002 | review | `-` | - | - | viscosity of an unlabelled substance | Pa*s, 20 degC 清水动力粘度 |
| 721 | `backend/physics/drydocking.py:39` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 水密度 [t/m³] |
| 722 | `backend/physics/flow_noise/farfield.py:51` | `AIR_RHO` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 723 | `backend/physics/flow_noise/farfield.py:53` | `WATER_RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 724 | `backend/physics/flow_noise/plate_modal.py:58` | `STEEL_RHO` | 7850 | review | `-` | - | - | density of an unlabelled substance | [kg/m³] 密度 |
| 725 | `backend/physics/flow_noise/plate_modal.py:61` | `AL_RHO` | 2700 | review | `-` | - | - | density of an unlabelled substance |  |
| 726 | `backend/physics/flow_noise/radiation.py:49` | `AIR_RHO` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | [kg/m³] |
| 727 | `backend/physics/flow_noise/radiation.py:51` | `WATER_RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] |
| 728 | `backend/physics/grain_stability/models.py:12` | `RHO_SW` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 729 | `backend/physics/grease_ice/ledger.py:23` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水密度 (Baltic 3.5 psu 近似) |
| 730 | `backend/physics/grease_ice/ledger.py:24` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 731 | `backend/physics/grease_ice/ledger.py:25` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s, ITTC-57 标准海水运动粘度 (15 degC) |
| 732 | `backend/physics/grease_ice/resistance.py:43` | `mu_creep` | 67.4 | review | `-` | - | - | viscosity of an unlabelled substance | Pa s, 颗粒蠕变黏度（由 Riska C2 传递） |
| 733 | `backend/physics/grounding.py:41` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 水密度 [t/m³] |
| 734 | `backend/physics/grounding.py:42` | `rho_seabed` | 1.8 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 海底密度 [t/m³] (沙质) |
| 735 | `backend/physics/holtrop/mennen.py:66` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 (15 degC) |
| 736 | `backend/physics/holtrop/mennen.py:67` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s  (15 degC, ITTC 1957 相关曲线) |
| 737 | `backend/physics/hull_form_opt/parametrics.py:56` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 738 | `backend/physics/hull_form_opt/parametrics.py:57` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s (15 °C) |
| 739 | `backend/physics/hull_girder_longterm.py:54` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 740 | `backend/physics/hull_hydroelastic/added_mass2d.py:55` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 741 | `backend/physics/hull_hydroelastic/added_mass3d.py:34` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 742 | `backend/physics/hull_opt/objectives.py:24` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 743 | `backend/physics/hull_opt/objectives.py:25` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s |
| 744 | `backend/physics/hull_roughness/townsin.py:53` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 745 | `backend/physics/hull_roughness/townsin.py:55` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 746 | `backend/physics/hull_vibration/sections.py:30` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 747 | `backend/physics/hull_vibration/sections.py:32` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | kg/m³ |
| 748 | `backend/physics/hydrofoil/hydrofoil_chain.py:24` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 749 | `backend/physics/hydrofoil/hydrofoil_ship.py:35` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 750 | `backend/physics/hydrofoil/hydrofoil_ship.py:36` | `RHO_FRESHWATER` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 淡水密度 [kg/m³] |
| 751 | `backend/physics/ice_astern/geometry.py:130` | `DEFAULT_RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | 冰密度 [kg/m^3] |
| 752 | `backend/physics/ice_astern/reverse_open_water.py:56` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 753 | `backend/physics/ice_breach_flooding/flooding.py:39` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 754 | `backend/physics/ice_channel/fsicr.py:61` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s, ITTC-57 标准海水运动粘度 (15 degC) |
| 755 | `backend/physics/ice_channel/fsicr.py:62` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 756 | `backend/physics/ice_channel/fsicr.py:63` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 757 | `backend/physics/ice_class_opt/objectives.py:35` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | Seawater density [kg/m³] |
| 758 | `backend/physics/ice_class_opt/objectives.py:36` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | Ice density [kg/m³] |
| 759 | `backend/physics/ice_class_opt/objectives.py:37` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | Kinematic viscosity [m²/s] (15°C seawater) |
| 760 | `backend/physics/ice_class_opt/objectives.py:40` | `MU_ICE` | 0.1 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | Ice-hull friction coefficient |
| 761 | `backend/physics/ice_energy/probes.py:43` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m^3, 海冰密度（与 API IceParams 默认一致） |
| 762 | `backend/physics/ice_fatigue/dynamic_ice_load.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] |
| 763 | `backend/physics/ice_fatigue/dynamic_ice_load.py:46` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | [kg/m³] |
| 764 | `backend/physics/ice_fatigue/dynamic_ice_load.py:55` | `DEFAULT_ICE_FLEXURAL_STRENGTH_PA` | 500000 | review | `-` | - | - | unclassified | 500 kPa |
| 765 | `backend/physics/ice_fatigue/fatigue_life.py:69` | `ice_flexural_strength_pa` | 500000 | review | `-` | - | - | unclassified |  |
| 766 | `backend/physics/ice_fatigue/fatigue_life.py:82` | `young_modulus_pa` | 2.069e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 767 | `backend/physics/ice_fatigue/fatigue_life.py:83` | `yield_strength_pa` | 3.55e+08 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 768 | `backend/physics/ice_fatigue/ice_load.py:32` | `STEEL_DENSITY_KG_M3` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 769 | `backend/physics/ice_fatigue/ice_load.py:34` | `WATER_DENSITY_KG_M3` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 770 | `backend/physics/ice_fatigue/ice_load.py:50` | `crush_strength_pa` | 2e+06 | review | `-` | - | - | unclassified |  |
| 771 | `backend/physics/ice_fatigue/stress_response.py:47` | `young_modulus_pa` | 2.069e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 772 | `backend/physics/ice_fatigue/stress_response.py:48` | `yield_strength_pa` | 3.55e+08 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 773 | `backend/physics/ice_fatigue/stress_response.py:242` | `rho_steel` | 7850 | review | `-` | - | - | density of an unlabelled substance | kg/m³ |
| 774 | `backend/physics/ice_friction/tests/test_ice_friction.py:88` | `mu_0` | 0.1 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 775 | `backend/physics/ice_interaction/lindqvist.py:76` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 776 | `backend/physics/ice_interaction/lindqvist.py:77` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 777 | `backend/physics/ice_lindqvist_anchor.py:61` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m³  (锚件 RHO_ICE, Timco & Weeks 2010) |
| 778 | `backend/physics/ice_lindqvist_anchor.py:64` | `MU_HULL_ICE` | 0.15 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | -      (锚件 MU_HULL_ICE, Riska 1997) |
| 779 | `backend/physics/ice_load/point_spectrum.py:50` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | [kg/m³] 海冰密度 |
| 780 | `backend/physics/ice_load/point_spectrum.py:51` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] 海水密度 |
| 781 | `backend/physics/ice_maneuvering/core.py:51` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 782 | `backend/physics/ice_maneuvering/core.py:52` | `mu_ice` | 0.15 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) |  |
| 783 | `backend/physics/ice_maneuvering/core.py:54` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 784 | `backend/physics/ice_maneuvering/core.py:132` | `nu_ice` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 785 | `backend/physics/ice_maneuvering/core.py:191` | `nu_ice` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) |  |
| 786 | `backend/physics/ice_maneuvering/ice_loads.py:95` | `density` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 787 | `backend/physics/ice_maneuvering/mmg.py:36` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水密度 |
| 788 | `backend/physics/ice_oil_spill/drift.py:49` | `oil_density` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 789 | `backend/physics/ice_oil_spill/drift.py:50` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 790 | `backend/physics/ice_oil_spill/drift.py:51` | `ice_density` | 910 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 791 | `backend/physics/ice_oil_spill/pipeline.py:107` | `oil_density` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 792 | `backend/physics/ice_oil_spill/pipeline.py:108` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 793 | `backend/physics/ice_oil_spill/pipeline.py:109` | `ice_density` | 910 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 794 | `backend/physics/ice_oil_spill/recovery.py:59` | `oil_density` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 795 | `backend/physics/ice_oil_spill/spreading.py:53` | `oil_density` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | kg/m^3, typical crude oil |
| 796 | `backend/physics/ice_oil_spill/spreading.py:54` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, seawater |
| 797 | `backend/physics/ice_oil_spill/spreading.py:56` | `water_viscosity` | 1.5e-06 | review | `-` | - | - | viscosity of an unlabelled substance | m^2/s (cold water ~5°C) |
| 798 | `backend/physics/ice_oil_spill/under_ice_lens.py:39` | `oil_density` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 799 | `backend/physics/ice_oil_spill/under_ice_lens.py:40` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 800 | `backend/physics/ice_oil_spill/under_ice_lens.py:41` | `ice_density` | 910 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | kg/m^3 |
| 801 | `backend/physics/ice_oil_spill/under_ice_lens.py:45` | `oil_viscosity` | 0.1 | review | `-` | - | - | viscosity of an unlabelled substance | Pa·s (crude oil at ~10°C) |
| 802 | `backend/physics/ice_oil_spill/under_ice_lens.py:46` | `water_viscosity` | 1.5e-06 | review | `-` | - | - | viscosity of an unlabelled substance | m^2/s |
| 803 | `backend/physics/ice_plate_bending.py:49` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m^3] |
| 804 | `backend/physics/ice_plate_bending.py:52` | `NU_ICE` | 0.3 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 |
| 805 | `backend/physics/ice_route_opt/fuel_model.py:19` | `fuel_density` | 0.95 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | Fuel density (kg/L), HFO typical |
| 806 | `backend/physics/ice_route_opt/resistance_model.py:32` | `flexural_strength` | 500000 | review | `-` | - | - | unclassified | Pa |
| 807 | `backend/physics/ice_shaft_whirl/ice_load.py:115` | `rho_i` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 808 | `backend/physics/ice_shm/constants.py:13` | `DEFAULT_YOUNG_MODULUS_PA` | 2.069e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 809 | `backend/physics/ice_thickness_inversion/__init__.py:81` | `ICE_FLEXURAL_STRENGTH` | 500000 | review | `-` | - | - | unclassified | σ_f [Pa] |
| 810 | `backend/physics/ice_thickness_inversion/__init__.py:82` | `ICE_DENSITY` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | ρ_ice [kg/m³] |
| 811 | `backend/physics/ice_thickness_inversion/__init__.py:83` | `ICE_YOUNG` | 5e+09 | review | `-` | - | - | elastic modulus, material not identified | E_ice [Pa] |
| 812 | `backend/physics/ice_thickness_inversion/__init__.py:84` | `WATER_DENSITY` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | ρ_w [kg/m³] |
| 813 | `backend/physics/ice_uq_mc/inputs.py:199` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 814 | `backend/physics/ice_uq_mc/models.py:46` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) |  |
| 815 | `backend/physics/ice_uq_resistance.py:183` | `nu_ice` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | Poisson's ratio |
| 816 | `backend/physics/ice_uq_resistance.py:185` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | seawater density (kg/m³) |
| 817 | `backend/physics/ice_uq_resistance.py:197` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | ice density (kg/m³) |
| 818 | `backend/physics/ice_uq_system/components.py:132` | `brash_mu_nominal` | 0.15 | review | `-` | - | - | viscosity of an unlabelled substance | 用于 C_ice 参数传递的名义摩擦系数 |
| 819 | `backend/physics/ice_uq_system/components.py:138` | `mu_ice_ice` | 0.6 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | 冰-冰摩擦（DECLARED，ISO 19906 常用 0.5~0.8） |
| 820 | `backend/physics/ice_vibration.py:272` | `ice_density` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | ρ_ice (kg/m^3) |
| 821 | `backend/physics/ice_vp_stress/anchors.py:28` | `E_YIELD` | 2 | review | `-` | - | - | unclassified |  |
| 822 | `backend/physics/ice_vp_stress/anchors.py:32` | `C_STRENGTH` | 20 | review | `-` | - | - | unclassified |  |
| 823 | `backend/physics/ice_vp_stress/anchors.py:36` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3, 海冰密度（CICE 默认 / Hibler 0.90e3 量级） |
| 824 | `backend/physics/ice_vp_stress/anchors.py:37` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水密度 |
| 825 | `backend/physics/ikeda_roll_damping.py:57` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ (海水) |
| 826 | `backend/physics/ikeda_roll_damping.py:58` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 827 | `backend/physics/ikeda_roll_damping.py:59` | `NU_WATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s (海水运动粘度, 15°C) |
| 828 | `backend/physics/ittc78_correlation.py:61` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 829 | `backend/physics/ittc78_correlation.py:62` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s (15°C) |
| 830 | `backend/physics/launching/bonjean.py:25` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m³] 海水 |
| 831 | `backend/physics/lifetime_degradation/resistance_evolution.py:77` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 832 | `backend/physics/lng_reliquefaction/core.py:64` | `LNG_DENSITY` | 450 | review | `-` | - | - | density of an unlabelled substance | kg/m³ |
| 833 | `backend/physics/lng_rollover/make_artifacts.py:38` | `LNG_DENSITY` | 450 | review | `-` | - | - | density of an unlabelled substance |  |
| 834 | `backend/physics/lng_sloshing.py:36` | `RHO_LNG` | 426 | review | `-` | - | - | density of an unlabelled substance | [kg/m^3] LNG 密度（甲烷为主，-162°C） |
| 835 | `backend/physics/lng_sloshing.py:37` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] 海水密度（参考） |
| 836 | `backend/physics/load_strength_core.py:121` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 837 | `backend/physics/mooring/env_loads.py:25` | `air_density` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 [kg/m^3] |
| 838 | `backend/physics/mooring/env_loads.py:33` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 [kg/m^3] |
| 839 | `backend/physics/mooring/env_loads.py:46` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 840 | `backend/physics/mooring_dp/capability.py:31` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 841 | `backend/physics/mooring_dp/loads.py:39` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 842 | `backend/physics/mooring_dp/loads.py:40` | `RHO_A` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 843 | `backend/physics/mooring_dp/thruster.py:36` | `_RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 844 | `backend/physics/mooring_drift/drift_force.py:55` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 845 | `backend/physics/panel_flow_core.py:47` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³（海水） |
| 846 | `backend/physics/parametric_hulls/hydrostatics.py:22` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, seawater density |
| 847 | `backend/physics/parametric_hulls/hydrostatics.py:23` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m^3, fresh water density |
| 848 | `backend/physics/parametric_hulls/hydrostatics.py:25` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s, kinematic viscosity of seawater at 15°C |
| 849 | `backend/physics/parametric_roll_ice/ice_damping.py:59` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 850 | `backend/physics/parametric_roll_ice/ice_damping.py:60` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 851 | `backend/physics/pipe_laying/hogbend.py:133` | `y_yield` | 0 | review | `-` | - | - | unclassified |  |
| 852 | `backend/physics/pipelay_slay/strain.py:45` | `rho_steel` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 853 | `backend/physics/pipelay_slay/strain.py:46` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 854 | `backend/physics/pipelay_slay/strain.py:50` | `internal_content_density` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 铺设时通常空管（空气≈0） |
| 855 | `backend/physics/preswirl_stator/envelope.py:46` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 856 | `backend/physics/preswirl_stator/lifting_line.py:36` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 857 | `backend/physics/preswirl_stator/wake.py:29` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 858 | `backend/physics/preswirl_stator/wake.py:46` | `swirl_strength` | 0.15 | review | `-` | - | - | unclassified | S = v_theta(R_prop)/V |
| 859 | `backend/physics/prop_closure/hull.py:73` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 860 | `backend/physics/prop_gap/bpf.py:36` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 861 | `backend/physics/prop_gap/lifting_line.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 862 | `backend/physics/prop_gap/lifting_line.py:46` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 863 | `backend/physics/prop_rudder/mmg.py:35` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 864 | `backend/physics/prop_rudder/rudder.py:26` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 865 | `backend/physics/prop_rudder/wake.py:44` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 866 | `backend/physics/propeller/inception.py:40` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 867 | `backend/physics/propeller_emergence.py:34` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 868 | `backend/physics/propeller_open_water/definitions.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, ITTC 海水 |
| 869 | `backend/physics/propeller_ventilation/thrust_model.py:61` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 870 | `backend/physics/propulsion/rudder_inflow.py:305` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 871 | `backend/physics/propulsion/wageningen.py:428` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 872 | `backend/physics/propulsion/wageningen.py:488` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 873 | `backend/physics/rao_validation/hull.py:82` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 (kg/m^3) |
| 874 | `backend/physics/resistance/kpi_res_5_10.py:39` | `RHO_MODEL` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 875 | `backend/physics/resistance/kpi_res_5_10.py:40` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 876 | `backend/physics/resistance/kpi_res_5_10.py:41` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 877 | `backend/physics/resistance_additivity/models.py:40` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | 海水运动粘度 (m²/s, 15°C) |
| 878 | `backend/physics/resistance_additivity/models.py:41` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 (kg/m³) |
| 879 | `backend/physics/resistance_audit/audit.py:244` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 880 | `backend/physics/restricted_maneuvering.py:51` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 881 | `backend/physics/restricted_water/interaction.py:35` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 882 | `backend/physics/restricted_water/panel_solver.py:54` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 883 | `backend/physics/restricted_water_interaction.py:51` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 884 | `backend/physics/ridge_ice/anchors.py:22` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水（近岸/北极上表层量级） |
| 885 | `backend/physics/ridge_ice/anchors.py:23` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3 海冰密度 |
| 886 | `backend/physics/ridge_ice/resistance.py:68` | `mu_r` | 0.6 | review | `-` | - | - | viscosity of an unlabelled substance | 龙骨碎石推挤摩擦/形状系数 |
| 887 | `backend/physics/ridge_ice/resistance.py:69` | `mu_s` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 脊帆清除系数 |
| 888 | `backend/physics/roll_damping/hull.py:31` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度（默认，可覆盖） |
| 889 | `backend/physics/roll_damping/sgisc_level2.py:261` | `nu_ref` | 0.5 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 890 | `backend/physics/roll_damping_speed.py:57` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 891 | `backend/physics/roll_damping_speed.py:563` | `mu_min` | 0.03 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 892 | `backend/physics/roughness/report.py:42` | `NU_SW` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 893 | `backend/physics/roughness_fouling.py:31` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 894 | `backend/physics/roughness_fouling.py:32` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 895 | `backend/physics/rudder/cavitation.py:26` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 896 | `backend/physics/rudder/flutter.py:52` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 897 | `backend/physics/rudder/polar.py:33` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s at 15 degC (ITTC-78 fresh/sea water value) |
| 898 | `backend/physics/rudder/polar.py:34` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 899 | `backend/physics/rudder_hydro/torque.py:90` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 900 | `backend/physics/rudder_hydrodynamics/cavitation.py:81` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 901 | `backend/physics/rudder_hydrodynamics/rudder.py:45` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 902 | `backend/physics/rudder_hydrodynamics.py:96` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] 海水密度 |
| 903 | `backend/physics/sea_chest_rudder/anchors.py:35` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 极地海水 (LITERATURE: 3.5 psu, 0 degC 近似) |
| 904 | `backend/physics/sea_chest_rudder/anchors.py:36` | `RHO_ICE` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m^3, 卤水/柱状冰常见值 (LITERATURE) |
| 905 | `backend/physics/sea_trial/fujiwara.py:56` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | [kg/m^3] standard air density at sea level |
| 906 | `backend/physics/sea_trial/iso15016_engine.py:25` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 907 | `backend/physics/sea_trial/iso15016_engine.py:26` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 908 | `backend/physics/sea_trial/ittc78.py:96` | `nu_15` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | [m^2/s] |
| 909 | `backend/physics/sea_trial/stawave2.py:44` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 910 | `backend/physics/seakeeping/stf_rao_validation.py:41` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 911 | `backend/physics/seakeeping_td/section.py:54` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 912 | `backend/physics/seakeeping_td/strip.py:57` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 913 | `backend/physics/seakeeping_validation/hulls.py:30` | `RHO_SW` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水 |
| 914 | `backend/physics/seakeeping_validation/hulls.py:31` | `RHO_FW` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 试验水池淡水 |
| 915 | `backend/physics/second_gen_stability/cpp.py:33` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 916 | `backend/physics/second_gen_stability/esd.py:44` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 917 | `backend/physics/second_gen_stability/surf_riding.py:55` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 918 | `backend/physics/second_gen_stability/surf_riding.py:82` | `mu_surge` | 0.06 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 919 | `backend/physics/second_order_drift.py:95` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 920 | `backend/physics/second_order_qtf_p45n07.py:71` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 921 | `backend/physics/second_order_wave_loads.py:73` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 922 | `backend/physics/selfprop_cavitation/coupled_propulsion.py:49` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m^3 (15 degC) |
| 923 | `backend/physics/selfprop_cavitation/freestreamline.py:46` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 924 | `backend/physics/selfprop_cavitation/sheet_cavity.py:39` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 925 | `backend/physics/sgisc_level2.py:52` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度（与 physics.holtrop.RHO_SEAWATER 同值） |
| 926 | `backend/physics/shaft_torsion/holzer.py:60` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance | 钢默认 |
| 927 | `backend/physics/shaft_torsional/shaft_model.py:33` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 (kg/m³) |
| 928 | `backend/physics/shaft_torsional/shaft_model.py:111` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 929 | `backend/physics/shafting_torsion/model.py:25` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 [kg/m³] |
| 930 | `backend/physics/shallow_maneuvering/bank_effect.py:57` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 931 | `backend/physics/shallow_maneuvering/mmg.py:67` | `RHO_SEA_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 932 | `backend/physics/shallow_squat_ukc.py:68` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 933 | `backend/physics/shallow_water/geometry.py:28` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 934 | `backend/physics/shallow_water/resistance.py:51` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 935 | `backend/physics/shallow_water/resistance.py:52` | `NU_W` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s (ITTC-15 salt water kinematic viscosity) |
| 936 | `backend/physics/shallow_water/resistance_amplification.py:38` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 937 | `backend/physics/shallow_water/resistance_amplification.py:39` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s (15°C) |
| 938 | `backend/physics/shallow_water_dual_ship.py:45` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 939 | `backend/physics/shallow_water_twin.py:604` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 |
| 940 | `backend/physics/shallow_water_twin.py:688` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 941 | `backend/physics/shore_power/assessment.py:38` | `RHO_SEA_WATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 942 | `backend/physics/slam_loads/chain.py:34` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 943 | `backend/physics/slam_loads/design_rules.py:33` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 944 | `backend/physics/slam_loads/slam_statistics.py:44` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 945 | `backend/physics/slam_loads/wedge_entry.py:61` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 946 | `backend/physics/slam_whip/wagner.py:43` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 947 | `backend/physics/slam_whipping.py:66` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 948 | `backend/physics/slam_whipping.py:78` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 949 | `backend/physics/slam_whipping.py:79` | `MU_AIR` | 1.8e-05 | migrate | `air/viscosity_dynamic` | 1.7893e-05 Pa*s | 0.6% | SI |  |
| 950 | `backend/physics/slamming/k3d_core.py:67` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 951 | `backend/physics/slamming/ledger.py:20` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 海水密度 (ITTC 标准海水 15 degC 近似) |
| 952 | `backend/physics/slamming/ledger.py:22` | `NU_STEEL` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 泊松比 (船用钢) |
| 953 | `backend/physics/slamming/ledger.py:24` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | kg/m^3, 钢密度 |
| 954 | `backend/physics/slamming/wagner.py:61` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 955 | `backend/physics/slamming.py:53` | `RHO_WATER` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 淡水密度 [kg/m³]（Zhao 实验用淡水） |
| 956 | `backend/physics/slamming_3d/baffle_dtn.py:51` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 957 | `backend/physics/slamming_3d/k3d.py:52` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 958 | `backend/physics/slamming_3d/spray_flux.py:45` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 959 | `backend/physics/slamming_3d/spray_flux.py:46` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 960 | `backend/physics/slamming_3d/two_legs.py:40` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 961 | `backend/physics/slamming_3d/wedge2d.py:46` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 海水 |
| 962 | `backend/physics/slamming_core.py:47` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |
| 963 | `backend/physics/slamming_core.py:48` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ 淡水密度（实验常用） |
| 964 | `backend/physics/slamming_whipping/chain.py:57` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 965 | `backend/physics/slamming_whipping_core.py:63` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 966 | `backend/physics/slamming_whipping_core.py:445` | `nu_plus` | 0 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 100.0% | SI |  |
| 967 | `backend/physics/sloshing/cases.py:72` | `rho_water_lu` | 0.8 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 968 | `backend/physics/sloshing/cases.py:73` | `rho_air_lu` | 0.4 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 969 | `backend/physics/sloshing/core.py:49` | `RHO_LNG` | 470 | review | `-` | - | - | density of an unlabelled substance | [kg/m^3] LNG 密度（工程带 430~470，取上限） |
| 970 | `backend/physics/sloshing/core.py:50` | `RHO_WATER` | 998.2 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | [kg/m^3] 水池模型试验（淡水） |
| 971 | `backend/physics/sloshing/fs_lbm.py:85` | `rho0` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 972 | `backend/physics/sloshing/nsw.py:39` | `RHO_LNG` | 470 | review | `-` | - | - | density of an unlabelled substance |  |
| 973 | `backend/physics/sloshing/tank_geometry.py:46` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 974 | `backend/physics/sloshing/tank_geometry.py:147` | `rho` | 425 | review | `-` | - | - | density of an unlabelled substance |  |
| 975 | `backend/physics/sloshing_baffle/perforated_baffle.py:63` | `rho` | 450 | review | `-` | - | - | density of an unlabelled substance | LNG 密度（kg/m^3） |
| 976 | `backend/physics/sloshing_tank/plate_fem.py:54` | `young` | 2.06e+11 | review | `-` | - | - | elastic modulus, material not identified |  |
| 977 | `backend/physics/sloshing_tank/plate_fem.py:55` | `poisson` | 0.3 | review | `-` | - | - | Poisson ratio, material not identified |  |
| 978 | `backend/physics/sloshing_tank/plate_fem.py:56` | `density` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 979 | `backend/physics/sloshing_tank/tank.py:95` | `density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | LNG 取 480 kg/m3；压载/水取 1025；默认水（保守） |
| 980 | `backend/physics/spectral_fatigue.py:38` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | seawater density [kg/m^3] |
| 981 | `backend/physics/stern_bearing/chain.py:74` | `density_steel` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 982 | `backend/physics/stern_bearing/thermal.py:28` | `mu_ref` | 0.05 | review | `-` | - | - | viscosity of an unlabelled substance | 参考温度下粘度 [Pa·s] |
| 983 | `backend/physics/stern_bearing/thermal.py:33` | `rho` | 870 | migrate | `sea_ice/density` | 900 kg/m^3 | 3.3% | SI; compared at T=0 degC (domain boundary) | 密度 [kg/m³] |
| 984 | `backend/physics/stern_tube_bearing/friction.py:27` | `RHO_OIL` | 890 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.1% | SI; compared at T=0 degC (domain boundary) | kg/m³ |
| 985 | `backend/physics/stern_tube_bearing/stribeck.py:52` | `rho` | 0.0005 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 峰元曲率半径 [m] |
| 986 | `backend/physics/stf_added_resistance.py:65` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ (海水) |
| 987 | `backend/physics/stf_added_resistance.py:66` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ (淡水，模型试验常用) |
| 988 | `backend/physics/structural_acoustics/cabin_noise.py:121` | `rho` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 989 | `backend/physics/structural_acoustics/coupling.py:128` | `rho` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 990 | `backend/physics/structural_acoustics/coupling.py:281` | `rho` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 991 | `backend/physics/structural_acoustics/cylindrical_shell.py:140` | `rho` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 992 | `backend/physics/structural_acoustics/cylindrical_shell.py:352` | `rho` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 993 | `backend/physics/structural_acoustics/plate_radiation.py:294` | `rho` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 994 | `backend/physics/torsional_vibration/__init__.py:41` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 [kg/m³] |
| 995 | `backend/physics/transom/transom_drag.py:74` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 996 | `backend/physics/two_phase_lbm.py:62` | `rho0` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | pseudo-potential scale in psi = rho0 (1 - exp(-rho |
| 997 | `backend/physics/two_phase_lbm.py:64` | `rho_init` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | initial uniform density for a homogeneous start |
| 998 | `backend/physics/underwater_acoustics.py:77` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] 海水 |
| 999 | `backend/physics/uq/resistance_case.py:85` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1000 | `backend/physics/ventilation/thermal_network.py:102` | `RHO_AIR` | 1.2 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 1001 | `backend/physics/vibration/dekker.py:52` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 [kg/m³] |
| 1002 | `backend/physics/vibration_core.py:44` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | kg/m³ |
| 1003 | `backend/physics/vibration_noise_core.py:65` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 1004 | `backend/physics/vibration_noise_core.py:66` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | [kg/m^3] |
| 1005 | `backend/physics/vibration_noise_core.py:68` | `NU_STEEL` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 泊松比 |
| 1006 | `backend/physics/wave_added_resistance/drift_force.py:56` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1007 | `backend/physics/wave_added_resistance/spectrum.py:36` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1008 | `backend/physics/wave_added_resistance/sta2.py:93` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 1009 | `backend/physics/wave_added_resistance.py:50` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 1010 | `backend/physics/wave_drift.py:72` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1011 | `backend/physics/wave_spectrum_core.py:56` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 1012 | `backend/physics/wave_speed_loss.py:146` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1013 | `backend/physics/weather_roll_fpk/weather.py:18` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 1014 | `backend/physics/wec_assist/point_absorber.py:54` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 1015 | `backend/physics/wind_assist/flettner.py:54` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1016 | `backend/physics/wind_load/core.py:15` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³, 标准空气密度 |
| 1017 | `backend/routing/weather_router.py:44` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 1018 | `backend/routing/weather_router.py:45` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 [kg/m³] |
| 1019 | `backend/services/damage_stability_trim.py:216` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1020 | `backend/services/digital_twin/models/seakeeping.py:115` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1021 | `backend/services/digital_twin/models/seakeeping.py:136` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1022 | `backend/services/dt_online/hull_physics.py:38` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 1023 | `backend/services/dt_online/hull_physics.py:39` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s  15 degC sea water |
| 1024 | `backend/services/ice_resistance_lindqvist.py:94` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m^3] |
| 1025 | `backend/services/ice_resistance_lindqvist.py:95` | `RHO_I` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | 海冰密度 [kg/m^3] |
| 1026 | `backend/services/ice_resistance_lindqvist.py:96` | `NU_ICE` | 0.33 | review | `sea_ice/viscosity_kinematic` | - | - | no database entry (UnknownProperty) | 海冰泊松比 [-] |
| 1027 | `backend/services/ice_resistance_lindqvist.py:100` | `MU_DEFAULT` | 0.05 | review | `-` | - | - | viscosity of an unlabelled substance | 冰-钢摩擦系数（MT Uikku 用值，0.05~0.15 带） |
| 1028 | `backend/services/iso19030.py:218` | `rho_air` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1029 | `backend/services/iso19030.py:219` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1030 | `backend/services/large_angle_gz.py:65` | `RHO_SEAWATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m^3, standard sea water (IS Code 2008 uses 1.025 |
| 1031 | `backend/services/lsa/launching_dynamics.py:23` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | seawater density [kg/m³] |
| 1032 | `backend/services/lsa/launching_dynamics.py:24` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | air density [kg/m³] |
| 1033 | `backend/services/natural_periods.py:87` | `RHO_SW` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, sea water |
| 1034 | `backend/services/occ_carbon_capture.py:59` | `RHO_SOL_DEFAULT` | 1010 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 1.1% | SI | kg/m^3 (30 wt% MEA, 40 C 量级) |
| 1035 | `backend/services/polar/ice_field.py:185` | `ridge_density_per_km` | 0 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1036 | `backend/services/propeller_hull_interaction.py:79` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 15 degC（与 Holtrop 主链一致） |
| 1037 | `backend/services/seakeeping.py:184` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1038 | `backend/services/seakeeping_flow/hulls.py:47` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1039 | `backend/services/seakeeping_flow/strip.py:58` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1040 | `backend/services/seakeeping_stats.py:57` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1041 | `backend/services/seakeeping_stats.py:299` | `NU_IN_DOMAIN` | 0.05 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 1042 | `backend/services/seakeeping_td/memory.py:128` | `mu_ratio` | 4 | review | `-` | - | - | dimensionless/derived coefficient |  |
| 1043 | `backend/services/seakeeping_td/time_domain.py:57` | `mu_ratio` | 4 | review | `-` | - | - | dimensionless/derived coefficient |  |
| 1044 | `backend/services/seakeeping_td/wagner.py:62` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1045 | `backend/services/sr_rudder_optimizer.py:47` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1046 | `backend/services/sr_rudder_optimizer.py:48` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1047 | `backend/services/wave_maneuvering/section_model.py:46` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1048 | `backend/services/weather_routing_cii.py:57` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | [kg/m^3] |
| 1049 | `backend/services/weather_routing_cii.py:58` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | [kg/m^3] |
| 1050 | `backend/structure/fracture_arrest/steel_grades.py:37` | `density` | 7850 | review | `-` | - | - | density of an unlabelled substance | Density [kg/m³] |
| 1051 | `backend/tests/test_aluminum_haz_buckling.py:36` | `NU_AL` | 0.33 | review | `-` | - | - | viscosity of an unlabelled substance |  |
| 1052 | `backend/tests/test_aluminum_haz_buckling.py:49` | `RHO_HAZ` | 0.62 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1053 | `backend/tests/test_aluminum_haz_buckling.py:384` | `expected_rho` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1054 | `backend/tests/test_appendage_decomposition.py:53` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1055 | `backend/tests/test_cavitation_bucket_p67n05.py:22` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1056 | `backend/tests/test_hullform_sensitivity_map_p200w1.py:106` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1057 | `backend/tests/test_hydrofoil_lifting_surface.py:64` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1058 | `backend/tests/test_lng_rollover.py:39` | `LNG_DENSITY` | 450 | review | `-` | - | - | density of an unlabelled substance |  |
| 1059 | `backend/tests/test_m6750_structure_chain.py:149` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1060 | `backend/tests/test_m6750_structure_chain.py:173` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1061 | `backend/tests/test_m6750_structure_chain.py:204` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1062 | `backend/tests/test_m6750_structure_chain.py:209` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1063 | `backend/tests/test_m6750_structure_chain.py:224` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1064 | `backend/tests/test_m6750_structure_chain.py:229` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1065 | `backend/tests/test_m6750_structure_chain.py:376` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1066 | `backend/tests/test_mmg_physics.py:527` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1067 | `backend/tests/test_propeller_bemt_cross_verification.py:44` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1068 | `backend/tests/test_seakeeping_dual_path_verification.py:30` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1069 | `backend/tests/test_slam_whipping_p305w2.py:19` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1070 | `backend/tests/test_wave_chain_m9411.py:38` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1071 | `backend/tools/added_resistance_audit.py:47` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1072 | `backend/verification/efd_source/sources.py:35` | `NU_NOMINAL` | 1.27e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 6.9% | SI |  |
| 1073 | `benchmarks/anchors/tools/wind_anchor.py:62` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³，与仓内 backend/api/coupled_maneuvering.py 一致 |
| 1074 | `benchmarks/ice_nav_operational_chain/build_artifacts.py:327` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 1075 | `benchmarks/verified/tshd_production/generate_artifacts.py:125` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1076 | `benchmarks/verified/tshd_production/generate_artifacts.py:280` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1077 | `benchmarks/verified/vibration_modal_verification.py:63` | `WATER_RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1078 | `c1794278110_555/repo/artifacts/EX1_tank_test_spec/make_audit_figs.py:24` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1079 | `docs/kpi_res_5_10/measure_kpi.py:60` | `RHO_SW` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1080 | `docs/kpi_res_5_10/measure_kpi.py:61` | `NU_SW` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 1081 | `docs/prereg/ex1_kcs/holtrop_mennen.py:57` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³, 15 °C 海水 |
| 1082 | `docs/prereg/ex1_kcs/holtrop_mennen.py:58` | `NU_SEA` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m²/s, 15 °C 海水运动粘度 |
| 1083 | `ice_whipping/ice_psd.py:66` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1084 | `ice_whipping/vessel.py:24` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1085 | `ice_whipping/vessel.py:25` | `RHO_S` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1086 | `lbm_debug_v10_stable.py:22` | `rho_phys` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1087 | `lbm_debug_v9.py:22` | `rho_phys` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1088 | `lbm_poiseuille_test.py:84` | `rho_out` | 1 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1089 | `lbm_suboff_les_v11.py:22` | `rho_phys` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1090 | `michell/artifacts/gen_p98_n03.py:25` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1091 | `michell/artifacts/generate_catamaran_validation.py:36` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1092 | `michell/cat_pipeline.py:40` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1093 | `michell/core.py:41` | `RHO_DEFAULT` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1094 | `michell/holtrop.py:18` | `NU_DEFAULT` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI |  |
| 1095 | `michell/powerlaw.py:33` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1096 | `michell/sensitivity.py:61` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1097 | `scripts/anchors/anchor_side_checks.py:72` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1098 | `scripts/gen_cavitation_inception_figures.py:145` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1099 | `scripts/gen_ex1_tank_test_spec.py:37` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m3 海水（母型） |
| 1100 | `scripts/gen_multihull_artifacts.py:37` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1101 | `scripts/generate_m6750_figures.py:102` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1102 | `scripts/m7864_figures.py:22` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1103 | `scripts/make_slamming_figures.py:31` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1104 | `scripts/maneuvering/mmg_engine_factors.py:21` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1105 | `scripts/p129w3_n08_seakeeping_chain.py:39` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1106 | `scripts/restricted_water_envelope_scan.py:46` | `RHO` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1107 | `scripts/run_p167w2_n06_inclining3d.py:46` | `RHO` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | sea water, t/m3 |
| 1108 | `scripts/run_rudder_hydro_validation.py:46` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1109 | `scripts/run_turning35_validation.py:46` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1110 | `scripts/verify_seakeeping_dual_path.py:61` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1111 | `scripts/verify_slamming.py:31` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1112 | `scripts/w22_ittc78_chain/run_chain.py:37` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1113 | `scripts/w22_ittc78_chain/run_chain.py:392` | `nu_m` | 1.004e-06 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 11.8% | SI | 20 °C 淡水（模型试验常规水温） |
| 1114 | `scripts/w25/holtrop_cb_envelope.py:59` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, 15 degC |
| 1115 | `scripts/wind_loads_p48w2b_n04.py:36` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1116 | `src/cavitation_erosion.py:35` | `RHO_WATER` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ 密度 |
| 1117 | `src/cavitation_erosion.py:37` | `MU_WATER` | 0.001002 | review | `-` | - | - | viscosity of an unlabelled substance | Pa·s 动力黏度 |
| 1118 | `src/cavitation_inception.py:45` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水 |
| 1119 | `src/cavitation_inception.py:46` | `RHO_FW` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³ 淡水 |
| 1120 | `src/crp.py:21` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1121 | `src/electric_propulsion.py:69` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 1122 | `src/emergency_stopping.py:24` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1123 | `src/energy_chain/ledger.py:35` | `RHO_WATER` | 990 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.9% | SI | kg/m³ |
| 1124 | `src/energy_chain/system.py:90` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1125 | `src/fire_smoke/solver.py:41` | `rho` | 1.2 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1126 | `src/fire_smoke/solver.py:46` | `soot_yield` | 0.024 | review | `-` | - | - | unclassified | 丙烷发烟率 [kg/kg]（文献 0.02~0.06） |
| 1127 | `src/fire_smoke/solver.py:56` | `nu_t_max` | 0.04 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) | 涡黏上限（显式扩散稳定域） |
| 1128 | `src/fuel/methanol_properties.py:41` | `RHO_REF` | 791.8 | review | `-` | - | - | density of an unlabelled substance | 20°C 时密度 [kg/m³] |
| 1129 | `src/generate_m6466_artifacts.py:46` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1130 | `src/ice/thermal_mechanical/params.py:26` | `RHO_ICE` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) | kg/m3  bulk sea ice             # TEXTBOOK |
| 1131 | `src/machinery_noise_iso6954.py:250` | `rho_c` | 400 | review | `-` | - | - | density of an unlabelled substance | 空气特性阻抗 [Pa·s/m] |
| 1132 | `src/polar_acceptance.py:27` | `rho_ice` | 917 | migrate | `sea_ice/density` | 900 kg/m^3 | 1.9% | SI; compared at T=0 degC (domain boundary) | kg/m³, ice density |
| 1133 | `src/polar_acceptance.py:28` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³, seawater density |
| 1134 | `src/polar_acceptance.py:111` | `mu_hull_ice` | 0.05 | review | `sea_ice/viscosity_dynamic` | - | - | no database entry (UnknownProperty) | Typical hull-ice friction coefficient (INDEPENDENT |
| 1135 | `src/polar_acceptance.py:174` | `mu_valanto` | 0.06 | review | `-` | - | - | viscosity of an unlabelled substance | Independent friction coefficient |
| 1136 | `src/propeller_emergence.py:52` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 1137 | `src/propeller_engine_matching.py:168` | `water_density` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | rho [kg/m^3] |
| 1138 | `src/propulsion/__init__.py:20` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1139 | `src/propulsion.py:20` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1140 | `src/propulsion_decoupling.py:20` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1141 | `src/propulsion_efficiency.py:37` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1142 | `src/propulsion_model.py:29` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 1143 | `src/propulsion_physics.py:51` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m³ |
| 1144 | `src/self_propulsion.py:57` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 kg/m^3 |
| 1145 | `src/ship_perf/air_lubric.py:52` | `RHO_SEAWATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3, standard seawater density |
| 1146 | `src/ship_perf/air_lubric.py:53` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 |
| 1147 | `src/ship_perf/air_lubric.py:55` | `NU_SEAWATER` | 1.19e-06 | migrate | `sea_water/viscosity_kinematic` | 1.1883e-06 m^2/s | 0.1% | SI | m^2/s, kinematic viscosity at 15°C |
| 1148 | `src/ship_perf/propeller_ice_load.py:93` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水 |
| 1149 | `src/ship_perf/wave_added.py:93` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m^3] |
| 1150 | `src/ship_platform/dredging/pump_matching.py:22` | `RHO_W` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1151 | `src/ship_platform/dredging/slurry_transport.py:40` | `RHO_W` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | water density [kg/m³] |
| 1152 | `src/ship_platform/dredging/slurry_transport.py:41` | `NU_W` | 1e-06 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 12.2% | SI | kinematic viscosity of water [m²/s] (20°C) |
| 1153 | `src/ship_platform/magnetics/dipole_field.py:33` | `MU_0_OVER_4PI` | 1e-07 | conflict | `fresh_water/viscosity_kinematic` | 1.1386e-06 m^2/s | 91.2% | SI | μ₀/(4π) [T·m/A] |
| 1154 | `src/ship_platform/maneuvering/wave_maneuvering.py:78` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1155 | `src/ship_platform/manoeuvring/mmg.py:54` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 (kg/m³) |
| 1156 | `src/ship_platform/vibration/machinery_noise.py:205` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 (kg/m³) |
| 1157 | `src/ship_platform/vibration/machinery_noise.py:398` | `rho_0` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 (kg/m³) |
| 1158 | `src/ship_platform/vibration/machinery_noise.py:422` | `rho_0` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1159 | `src/stf_lewis_2d.py:54` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ |
| 1160 | `src/structural_acoustics/cylinder.py:35` | `RHO0_DEFAULT` | 1.204 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1161 | `src/structural_acoustics/plate.py:35` | `RHO0_DEFAULT` | 1.204 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 @20°C [kg/m³] |
| 1162 | `src/structural_acoustics/room.py:27` | `RHO0_DEFAULT` | 1.204 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1163 | `src/vibration_noise/cabin_acoustics.py:100` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 [kg/m³] |
| 1164 | `src/vibration_noise/cabin_acoustics.py:102` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 水密度 [kg/m³] |
| 1165 | `src/vibration_noise/propeller_excitation.py:30` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | 海水密度 [kg/m³] |
| 1166 | `src/waterjet.py:48` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³, seawater density |
| 1167 | `src/waterjet.py:49` | `RHO_FRESH` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | kg/m³, freshwater density |
| 1168 | `src/weather_routing/model.py:77` | `current_strength` | 0 | review | `-` | - | - | unclassified |  |
| 1169 | `src/weather_routing/ship.py:26` | `RHO_W` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1170 | `src/weather_routing/ship.py:27` | `RHO_A` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1171 | `tests/acoustics/test_helmholtz_bem.py:32` | `RHO` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1172 | `tests/is2008_clause_matrix.py:70` | `RHO_T_M3` | 1.025 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) |  |
| 1173 | `tests/references_seakeeping.py:52` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | sea-water density used by the platform |
| 1174 | `tests/ship/test_propulsion_closure.py:34` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1175 | `tests/ship/test_slamming_wedge_validation.py:300` | `yield_stress` | 235 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 1176 | `tests/ship/test_slamming_wedge_validation.py:314` | `yield_stress` | 235 | review | `-` | - | - | yield stress, steel grade not identified in the code |  |
| 1177 | `tests/stability_analytic_benchmark.py:42` | `RHO_T_M3` | 1.025 | review | `-` | - | - | turbulent/sub-grid quantity (solver-own length scale) |  |
| 1178 | `tests/test_added_resistance_single_source.py:50` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1179 | `tests/test_added_resistance_verification.py:35` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1180 | `tests/test_air_resistance_P170W2N01.py:34` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 1181 | `tests/test_beam_wet_modal.py:43` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | kg/m^3 |
| 1182 | `tests/test_beam_wet_modal.py:44` | `RHO_WATER` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 1183 | `tests/test_beam_wet_modal.py:90` | `rho_A` | 100 | review | `-` | - | - | density of an unlabelled substance |  |
| 1184 | `tests/test_buckling_bc_maturity_m13978.py:429` | `rho` | 0.8 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1185 | `tests/test_bwm_d1_operation_window.py:45` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1186 | `tests/test_cavitation_erosion.py:39` | `rho` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1187 | `tests/test_cavitation_erosion.py:101` | `rho` | 998 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1188 | `tests/test_damage_stability_3dof_m7864.py:47` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1189 | `tests/test_damage_stability_full.py:51` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1190 | `tests/test_draught_survey.py:58` | `BOX_RHO` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m³ |
| 1191 | `tests/test_draught_survey.py:264` | `rho1` | 1.01 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1192 | `tests/test_draught_survey.py:265` | `rho2` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1193 | `tests/test_dredging_production.py:91` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1194 | `tests/test_dredging_production.py:110` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1195 | `tests/test_dredging_production.py:175` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1196 | `tests/test_dredging_production.py:191` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1197 | `tests/test_dredging_production.py:268` | `rho_m` | 1200 | review | `-` | - | - | density of an unlabelled substance |  |
| 1198 | `tests/test_dredging_production.py:494` | `rho_s` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1199 | `tests/test_electric_propulsion_noise_chain.py:70` | `rho` | 7800 | review | `-` | - | - | density of an unlabelled substance |  |
| 1200 | `tests/test_fem_module_verification.py:44` | `RHO` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1201 | `tests/test_fuel_processing.py:127` | `rho_p` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI | 水 |
| 1202 | `tests/test_fuel_processing.py:128` | `rho_f` | 980 | review | `-` | - | - | density of an unlabelled substance | HFO |
| 1203 | `tests/test_fuel_processing.py:161` | `hfo_density_kgm3` | 980 | review | `-` | - | - | density of an unlabelled substance |  |
| 1204 | `tests/test_fuel_processing.py:175` | `hfo_density_kgm3` | 980 | review | `-` | - | - | density of an unlabelled substance |  |
| 1205 | `tests/test_fuel_processing.py:189` | `nu_cSt` | 10 | review | `-` | - | - | viscosity of an unlabelled substance | 10 cSt |
| 1206 | `tests/test_fuel_processing.py:190` | `rho` | 980 | review | `-` | - | - | density of an unlabelled substance | kg/m³ |
| 1207 | `tests/test_hull_roughness_degradation_p173w2.py:48` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1208 | `tests/test_ice_astern_breaking.py:67` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1209 | `tests/test_inclining3d_asym_m9701.py:43` | `RHO` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1210 | `tests/test_inclining_experiment.py:31` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1211 | `tests/test_inclining_test.py:115` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1212 | `tests/test_inclining_test.py:161` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1213 | `tests/test_inclining_test.py:210` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1214 | `tests/test_inclining_test.py:275` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1215 | `tests/test_inclining_test.py:332` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1216 | `tests/test_inclining_test.py:403` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1217 | `tests/test_inclining_test.py:448` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1218 | `tests/test_inclining_test.py:565` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1219 | `tests/test_inclining_test.py:591` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1220 | `tests/test_inclining_test.py:627` | `rho_water` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1221 | `tests/test_kpi_qc_recheck.py:211` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1222 | `tests/test_kpi_qc_recheck.py:243` | `rho_water` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1223 | `tests/test_lng_sloshing.py:321` | `rho` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1224 | `tests/test_machinery_noise_iso6954.py:71` | `poisson_ratio` | 0.3 | review | `-` | - | - | dimensionless/derived coefficient |  |
| 1225 | `tests/test_moving_load_ice.py:38` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1226 | `tests/test_multihull.py:32` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1227 | `tests/test_plate_radiation_convention.py:31` | `RHO_AIR` | 1.21 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | 空气密度 kg/m³ |
| 1228 | `tests/test_plate_radiation_convention.py:32` | `RHO_STEEL` | 7850 | review | `-` | - | - | density of an unlabelled substance | 钢密度 kg/m³ |
| 1229 | `tests/test_plate_radiation_convention.py:34` | `NU_STEEL` | 0.3 | review | `-` | - | - | viscosity of an unlabelled substance | 钢泊松比 |
| 1230 | `tests/test_propeller_vibration_chain.py:63` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1231 | `tests/test_propeller_vibration_chain.py:257` | `rho_s` | 7850 | review | `-` | - | - | density of an unlabelled substance | kg/m³ |
| 1232 | `tests/test_propeller_vibration_chain.py:300` | `rho_s` | 7850 | review | `-` | - | - | density of an unlabelled substance |  |
| 1233 | `tests/test_resistance_ittc78.py:52` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1234 | `tests/test_resistance_ittc78.py:153` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1235 | `tests/test_restricted_water_shipship.py:50` | `RHO` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1236 | `tests/test_seakeeping_statistics_chain.py:34` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1237 | `tests/test_shaft_torsion.py:62` | `rho` | 7850 | review | `-` | - | - | density of an unlabelled substance | 密度 [kg/m³] |
| 1238 | `tests/test_shallow_water_twin.py:121` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1239 | `tests/test_slamming_3d.py:20` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1240 | `tests/test_sloshing_m8051.py:36` | `RHO` | 470 | review | `-` | - | - | density of an unlabelled substance |  |
| 1241 | `tests/test_speed_loss_in61_n01.py:239` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1242 | `tests/test_static_stability_shared_kernel.py:40` | `RHO` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) |  |
| 1243 | `tests/test_stawave2_verification.py:38` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1244 | `tests/test_stf_added_resistance.py:39` | `RHO_FRESH` | 1000 | migrate | `fresh_water/density` | 999.1 kg/m^3 | 0.1% | SI |  |
| 1245 | `tests/test_stf_added_resistance.py:40` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1246 | `tests/test_tshd_durand_hydraulics.py:19` | `RHO_S` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1247 | `tests/test_tshd_pump_hopper_production.py:20` | `RHO_S` | 2650 | review | `-` | - | - | density of an unlabelled substance |  |
| 1248 | `tests/test_vibration_modal_kernel.py:30` | `RHO_F` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1249 | `tests/test_w17_resistance_repair.py:52` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1250 | `tests/test_wave_added_resistance.py:32` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1251 | `tests/test_wave_speed_loss.py:31` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1252 | `tests/test_wave_speed_loss_p248n10.py:23` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1253 | `tests/test_wec_coupled_dynamics.py:27` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1254 | `tests/test_wetted_surface_single_source_m16886.py:49` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1255 | `tools/chain_gate/chain_gate.py:50` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m^3 |
| 1256 | `tools/seakeeping_truncation/run_matrix.py:39` | `RHO` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1257 | `tools/validation/verify_rudder_p40w3c_n06.py:203` | `rho` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1258 | `tools/verif/dynstab/reference.py:44` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m³ |
| 1259 | `tools/verif/is2008/analytic_forms.py:40` | `RHO_SEA_WATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m³，仅用于输入自洽的排水量 |
| 1260 | `tools/verif/is2008/weather_reference.py:55` | `RHO_SEA_WATER` | 1.025 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | t/m³，仅用于输入排水量自洽检查 |
| 1261 | `tools/verif/wind_load/wind_cases.py:44` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 (15 degC) — dynamic-pressure route only |
| 1262 | `tools/verif/wind_load/wind_model.py:43` | `RHO_AIR` | 1.225 | review | `-` | - | - | non-SI density unit (t/m^3 or g/cm^3) | kg/m^3 |
| 1263 | `uikku_vv/fsd.py:154` | `rho_i` | 900 | migrate | `sea_ice/density` | 900 kg/m^3 | 0.0% | SI; compared at T=0 degC (domain boundary) |  |
| 1264 | `uikku_vv/fsd.py:155` | `rho_w` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI |  |
| 1265 | `verified/propulsion_verification/propulsion_verification.py:35` | `RHO_SEA` | 1025 | migrate | `sea_water/density` | 1025.9 kg/m^3 | 0.1% | SI | kg/m³ 海水密度 |

## 结论

- 共 1265 处字面量物性/材料常量：`migrate` 800、`conflict` 12、`review` 453。
- 网格/格子单位与无量纲系数（LBM `nu_water=0.03`、相场 `rho_water=1.0`、湍流/亚格子黏性、Weibull 模数）**不迁移**：它们没有 SI 量纲，迁进物性库会制造单位错误；正确做法是在调用点旁注明有量纲 duty point。
- 有量纲且已标注介质的常量（`NU_WATER=1.19e-6`、`RHO_SW=1025`、`RHO_AIR=1.225` 等）与库值吻合（≤10%），可直接换成库查询，从而继承 ITTC / ISO / 船级社规范的出处。
- `conflict` 行多为「物质或温度不同」而非库错，例如淡水 20 °C 取值 与其他温度取值对比；按本表口径需 owner 逐个确认后迁移。
