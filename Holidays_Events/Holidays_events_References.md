
## 1. National / Religious / “Other” Public Holidays (2023)

These sources give the official dates and types of holidays I used in the CSV:

* **OfficeHolidays – National Holidays in Bangladesh in 2023**
  Lists all 2023 government and national holidays (Language Martyrs’ Day, Shab-e-Barat, Independence Day, Pohela Boishakh, Eid holidays, Labour Day, Buddha Purnima, Eid-ul-Adha, Ashura, National Mourning Day, Janmashtami, Eid-e-Miladunnabi, Victory Day, Christmas etc.).

  * Link: [https://www.officeholidays.com/countries/bangladesh/2023](https://www.officeholidays.com/countries/bangladesh/2023)

* **Timeanddate – Holidays and Observances in Bangladesh in 2023**
  Shows dates, names and types (Government holiday, Optional, Observance). I used it to cross-check OfficeHolidays and to include things like New Year’s Day, Bangabandhu Homecoming Day, etc.

  * Link: [https://www.timeanddate.com/holidays/bangladesh/2023](https://www.timeanddate.com/holidays/bangladesh/2023)

* **BD Visa / Bangladesh Deputy High Commission – “Holiday List 2023” (PDF)**
  A concrete PDF list used to verify specific dates for Shab-e-Qadar, Pohela Boishakh, Eid-ul-Fitr block (21–23 Apr), May Day, Buddha Purnima, Eid-ul-Azha block (28–30 Jun), Ashura, National Mourning Day, Janmashtami, Eid-e-Miladunnabi etc.

  * Link: [https://bdvisa.com/uploads/Holiday-List-2023.pdf](https://bdvisa.com/uploads/Holiday-List-2023.pdf)

* **VFS Global Public Holidays 2023 – Dhaka (PDF)**
  Another cross-check of dates where their visa centre was closed (e.g. 1 Jan, 21 Feb, 8 Mar, 26 Mar, 19 Apr, 4 May, 28 Jun).

  * Link: [https://assets.ctfassets.net/xxg4p8gt3sg6/39tfeGo0wDod7DrNVmM3NZ/783593878c1ee043f703da54238a79b7/Public_Holiday-Bangladesh-2023-EN.pdf](https://assets.ctfassets.net/xxg4p8gt3sg6/39tfeGo0wDod7DrNVmM3NZ/783593878c1ee043f703da54238a79b7/Public_Holiday-Bangladesh-2023-EN.pdf)

Those four are the **main sources for the “national_holiday”, “religious_holiday”, “bank_holiday” rows** in the CSV.

---

## 2. Cultural Festivals and Mass Religious Events

### Pohela Falgun (Spring festival, 14 Feb 2023)

* **Pohela Falgun – Wikipedia**
  Explains the festival and notes that, after calendar reforms, Pohela Falgun has been on **14 February** since 2020, coinciding with Valentine’s Day.

  * Link: [https://en.wikipedia.org/wiki/Pohela_Falgun](https://en.wikipedia.org/wiki/Pohela_Falgun)

* **Xinhua / English News – “People celebrate Pohela Falgun in Dhaka, Bangladesh” (14 Feb 2023)**
  Confirms that in 2023 people in Dhaka celebrated Pohela Falgun and Valentine’s Day together on **14 February 2023**.

  * Example link: [https://english.news.cn/asiapacific/20230214/82f398faa2f44f59a80129f3fa840581/c.html](https://english.news.cn/asiapacific/20230214/82f398faa2f44f59a80129f3fa840581/c.html)

* **Dhaka Tribune – “How Falgun and Valentine’s Day collide to create the most…” (14 Feb 2023)**
  Describes how both are celebrated on the same day and the commercial impact (flowers, gifts, etc.).

  * Link: [https://www.dhakatribune.com/bangladesh/304875/how-falgun-and-valentine%E2%80%99s-day-collide-to-create](https://www.dhakatribune.com/bangladesh/304875/how-falgun-and-valentine%E2%80%99s-day-collide-to-create)

These are behind the **Pohela Falgun + Valentine’s Day** line in the CSV.

---

### Bishwa Ijtema (January 2023, Tongi)

* **Bangladesh Sangbad Sangstha (BSS) – “First phase of Bishwa Ijtema begins Jan 13”**
  States that the **first phase** of Bishwa Ijtema 2023 ran **13–15 January**, and the **second phase** **20–22 January**, in Tongi near Dhaka.

  * Link: [https://www.bssnews.net/news-flash/104226](https://www.bssnews.net/news-flash/104226)

Those dates are the basis for the **two “Bishwa Ijtema Phase 1 / Phase 2” event rows**.

---

## 3. Ramadan 2023 (behavioural regime)

For the Ramadan-period row (I marked it as an approximate “start of Ramadan / regime change”):

* **Dhaka Tribune – “Sehri, Iftar timings for Ramadan 2023”**
  Islamic Foundation schedule: Ramadan expected to begin **24 March 2023**, subject to moon sighting.

  * Example link: [https://www.dhakatribune.com/bangladesh/305741/sehri-iftar-timings-for-ramadan-2023](https://www.dhakatribune.com/bangladesh/305741/sehri-iftar-timings-for-ramadan-2023)

* **Aladhan – Ramadan prayer times Dhaka 2023**
  Lists Ramadan days for Dhaka (1 Ramadan = 23 March 2023, depending on moon).

  * Link: [https://aladhan.com/ramadan-prayer-times/2023/Dhaka/Bangladesh](https://aladhan.com/ramadan-prayer-times/2023/Dhaka/Bangladesh)

I used this to treat **late March–April 2023** as a special “Ramadan consumption pattern” period.

---

## 4. Climate / Weather Data Sources

For the **monthly typical temperature and rainfall** (the columns `typical_max_temp_C`, `typical_min_temp_C`, `typical_rain_mm`):

* **Climate of Dhaka – Wikipedia (using Bangladesh Meteorological Department data)**

  * Contains the monthly climate table for Dhaka (mean max/min temp, rainfall, etc.).
  * Link: [https://en.wikipedia.org/wiki/Climate_of_Dhaka](https://en.wikipedia.org/wiki/Climate_of_Dhaka)

* **Climatestotravel – Dhaka climate** (cross-check for seasonal description: dry Nov–Mar, rainy May–Oct).

  * Link: [https://www.climatestotravel.com/climate/bangladesh/dhaka](https://www.climatestotravel.com/climate/bangladesh/dhaka)

* **Climate-data.org – Dhaka climate by month** (used to verify approximate numbers).

  * Link: [https://en.climate-data.org/asia/bangladesh/dhaka-division/dhaka-1062098/](https://en.climate-data.org/asia/bangladesh/dhaka-division/dhaka-1062098/)

For **daily weather data (if you want to re-build features)** I suggested:

* **Mendeley dataset – “High Volume Real-World Weather Data”**
  Daily rainfall, temperature, humidity & sunshine for **35 weather stations in Bangladesh up to 2023**, originally from Bangladesh Meteorological Department.

  * Link: [https://data.mendeley.com/datasets/tbrhznpwg9/1](https://data.mendeley.com/datasets/tbrhznpwg9/1)

* **Kaggle – Bangladesh Weather History (Dhaka City)**

  * Link: [https://www.kaggle.com/datasets/talhabu/bangladesh-weather-history](https://www.kaggle.com/datasets/talhabu/bangladesh-weather-history)

* **Kaggle – Bangladesh Weather Dataset (1901–2023)**

  * Link: [https://www.kaggle.com/datasets/yakinrubaiat/bangladesh-weather-dataset](https://www.kaggle.com/datasets/yakinrubaiat/bangladesh-weather-dataset)

I didn’t pull raw daily values directly into the CSV; instead I used **monthly normal values** as a compact summary and pointed you to these datasets for detailed daily joining.

---

## 5. Extreme Weather / Floods (August 2023, Chattogram region)

For the “monsoon floods & landslides August 2023” entry:

* **IFRC / ReliefWeb – “Bangladesh: Flash Flood and Landslides in Chattogram Region – Situation Report 2, 16 August 2023”**

  * Link: [https://reliefweb.int/report/bangladesh/bangladesh-flash-flood-and-landslides-chattogram-region-situation-report-2-16-august-2023](https://reliefweb.int/report/bangladesh/bangladesh-flash-flood-and-landslides-chattogram-region-situation-report-2-16-august-2023)

* **ADRC – Bangladesh: Flood 2023-08-07**

  * Link: [https://www.adrc.asia/view_disaster_en.php?Key=2632&Lang=en&NationCode=](https://www.adrc.asia/view_disaster_en.php?Key=2632&Lang=en&NationCode=)

* **NIRAPAD – Monthly Hazard Incidence Report August 2023**

  * Link: [https://www.nirapad.org.bd/public/assets/resource/monthlyHazard/1694582519_Monthly%20Hazard%20Incidence%20Report%20August_2023.pdf](https://www.nirapad.org.bd/public/assets/resource/monthlyHazard/1694582519_Monthly%20Hazard%20Incidence%20Report%20August_2023.pdf)

These show heavy rains, flash floods and landslides affecting Chattogram division in early–mid August 2023. I summarised that as a **regional weather shock window**.

---

## 6. Infrastructure Events

### Dhaka Elevated Expressway (Airport–Farmgate section)

* **Wikipedia – Dhaka Elevated Expressway** (mentions that the country’s first elevated expressway was opened on **2 September 2023**).

  * Link: [https://en.wikipedia.org/wiki/Dhaka_Elevated_Expressway](https://en.wikipedia.org/wiki/Dhaka_Elevated_Expressway)

* **The Daily Star – “Dhaka’s First Elevated Expressway: All set for grand opening” (2 Sept 2023)**

  * Link: [https://www.thedailystar.net/news/bangladesh/transport/news/dhakas-first-elevated-expressway-all-set-grand-opening-3408826](https://www.thedailystar.net/news/bangladesh/transport/news/dhakas-first-elevated-expressway-all-set-grand-opening-3408826)

* **Dhaka Tribune – “PM Hasina to inaugurate Dhaka Elevated Expressway on September 2”**

  * Link: [https://www.dhakatribune.com/bangladesh/322472/quader-pm-hasina-to-inaugurate-dhaka-elevated](https://www.dhakatribune.com/bangladesh/322472/quader-pm-hasina-to-inaugurate-dhaka-elevated)

These are the basis for the **“infrastructure_event: expressway opening on 2023-09-02”** row.

---

### Dhaka–Bhanga Railway via Padma Bridge (10 Oct 2023)

* **Railway Gazette – “Dhaka–Bhanga railway across the Padma Bridge opens”**
  Confirms **82 km line from Dhaka to Bhanga via Padma Bridge inaugurated 10 October 2023**.

  * Link: [https://www.railwaygazette.com/infrastructure/dhaka-bhanga-railway-across-the-padma-bridge-opens/65144.article](https://www.railwaygazette.com/infrastructure/dhaka-bhanga-railway-across-the-padma-bridge-opens/65144.article)

* **Xinhua – “BRI railway via Padma Bridge in Bangladesh all set for inauguration”**

  * Link: [https://english.news.cn/20230909/a7b49590f1254fe18d2d7c312da10c66/c.html](https://english.news.cn/20230909/a7b49590f1254fe18d2d7c312da10c66/c.html)

Those underpin the **“Padma rail opening 2023-10-10”** event row.

---

## 7. Political Rallies, Blockades and Strikes (Late Oct–Nov 2023)

For the **BNP rally and subsequent blockades / hartals**, I relied mainly on Reuters:

* **Reuters – “Bangladesh opposition protest turns violent amid calls for PM to resign” (28 Oct 2023)**
  Large BNP protest in Dhaka, one policeman killed, >100 injured.

  * Link: [https://www.reuters.com/world/asia-pacific/bangladesh-opposition-protest-turns-violent-amid-calls-pm-resign-2023-10-28/](https://www.reuters.com/world/asia-pacific/bangladesh-opposition-protest-turns-violent-amid-calls-pm-resign-2023-10-28/)

* **Reuters – “Opposition activists held over policeman’s death in Bangladesh protest” (29 Oct 2023)**
  Follow-up arrests after the same protest.

  * Link: [https://www.reuters.com/world/asia-pacific/opposition-activists-held-over-policemans-death-bangladesh-protest-2023-10-29/](https://www.reuters.com/world/asia-pacific/opposition-activists-held-over-policemans-death-bangladesh-protest-2023-10-29/)

* **Reuters – “Two killed in anti-government protest in Bangladesh” (31 Oct 2023)**
  Describes a **three-day blockade of roads** called by BNP from **31 October 2023**.

  * Link: [https://www.reuters.com/world/asia-pacific/two-killed-anti-government-protest-bangladesh-2023-10-31/](https://www.reuters.com/world/asia-pacific/two-killed-anti-government-protest-bangladesh-2023-10-31/)

I then simplified this into a broader **“blockade / protest regime” period** for modelling (rather than listing every single hartal day).

---

## 8. School / University Academic Year

For the **“academic year starts/ends in January/December”** info (those rows are tagged as approximate in the CSV):

* **Annual Primary School Census (APSC) – Directorate of Primary Education, Bangladesh**
  Multiple APSC reports explicitly say **“the school year begins in January and ends in December”**.

  * Example (APSC 2022 PDF): [https://dpe.portal.gov.bd/sites/default/files/files/dpe.portal.gov.bd/publications/d6bc8fcc_3e7a_415c_838e_65b640279012/APSC%202022_Final%20Report_31.05.23.pdf](https://dpe.portal.gov.bd/sites/default/files/files/dpe.portal.gov.bd/publications/d6bc8fcc_3e7a_415c_838e_65b640279012/APSC%202022_Final%20Report_31.05.23.pdf)

* **EPDC – National Education Profile: Bangladesh**
  Also states that the **academic year begins in January and ends in December**.

  * Link: [https://www.epdc.org/sites/default/files/documents/EPDC_NEP_2018_Bangladesh.pdf](https://www.epdc.org/sites/default/files/documents/EPDC_NEP_2018_Bangladesh.pdf)

These give the **structure** of the year. I did **not** have a single official line saying “exactly 1 January 2023 is the first day of class”, so in the CSV I **approximated** start/end for 2023 as `2023-01-01` / `2023-12-31` and labelled them “approx.”. For your analysis, you can replace those with precise dates from specific school calendars if needed.

Individual schools/universities publish their own detailed calendars (examples):

* **Scholastica School – School Year & Academic Calendar** (example of term structure July–June, with breaks).

  * Link: [https://cacdn03.freeli.io/scholastica/academics/school-calendar.php](https://cacdn03.freeli.io/scholastica/academics/school-calendar.php)

* **Individual university calendars** (e.g., DU, ULAB, etc.) – you can fetch exact semester start/end for specific institutions you care about.

---

## 9. Where the CSV “guesses” versus “direct copies”

To be fully transparent:

* **Directly supported by sources (dates & status):**
  All the **public holidays**, **Eid blocks**, **Pohela Boishakh**, **Buddha Purnima**, **Ashura**, **National Mourning Day**, **Janmashtami**, **Eid-e-Miladunnabi**, **Victory Day**, **Christmas**, plus **Bishwa Ijtema dates**, **expressway opening (2 Sept)**, **rail opening (10 Oct)**, and **protest/blockade dates**.

* **Modeling choices / approximations:**

  * Treating Ramadan as a **behavioural regime window** rather than one date (based on Islamic Foundation schedule).
  * Encoding **August 2023 Chattogram floods** as one “regional weather shock” period, summarising more detailed situation reports.
  * Using **monthly climate normals for Dhaka** as typical weather values, instead of day-by-day values.
  * Approximating **school academic start/end** as 1 Jan / 31 Dec based on official statements that the academic year *runs January–December*, not on a specific “first school day” notice.

If you tell me **which rows you care about most** (e.g. just national holidays and blockades, or also school calendars), I can help you:

* strip out approximations,
* plug in **exact day-level weather** from one of the datasets above, and
* produce a **sanitised CSV + Python join script** that is purely data-driven with explicit links back to each source.






















