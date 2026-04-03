"""
cities.py — World City Database
================================
Provides country → city mappings.
India: all major cities across all 28 states + 8 UTs
World: top cities for every country
"""

# ─────────────────────────────────────────────────────────────────────────────
# India — all states and UTs with their major cities
# ─────────────────────────────────────────────────────────────────────────────
INDIA_CITIES: dict[str, list[str]] = {
    "Andhra Pradesh":         ["Visakhapatnam","Vijayawada","Guntur","Nellore","Kurnool","Tirupati","Kakinada","Rajahmundry","Eluru","Ongole","Anantapur","Kadapa","Srikakulam","Vizianagaram","Bhimavaram"],
    "Arunachal Pradesh":      ["Itanagar","Naharlagun","Pasighat","Tezpur","Ziro"],
    "Assam":                  ["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon","Tinsukia","Tezpur","Bongaigaon","Sivasagar","Diphu"],
    "Bihar":                  ["Patna","Gaya","Bhagalpur","Muzaffarpur","Purnia","Darbhanga","Bihar Sharif","Arrah","Begusarai","Katihar","Munger","Chhapra","Hajipur","Sasaram","Jehanabad"],
    "Chhattisgarh":           ["Raipur","Bhilai","Bilaspur","Korba","Durg","Rajnandgaon","Jagdalpur","Ambikapur","Raigarh","Dhamtari"],
    "Goa":                    ["Panaji","Margao","Vasco da Gama","Mapusa","Ponda","Bicholim"],
    "Gujarat":                ["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar","Jamnagar","Junagadh","Gandhinagar","Anand","Nadiad","Morbi","Surendranagar","Bharuch","Valsad","Navsari","Mehsana","Patan","Gandhidham"],
    "Haryana":                ["Faridabad","Gurugram","Panipat","Ambala","Yamunanagar","Rohtak","Hisar","Karnal","Sonipat","Panchkula","Bhiwani","Sirsa","Bahadurgarh","Jind","Thanesar"],
    "Himachal Pradesh":       ["Shimla","Dharamshala","Solan","Mandi","Baddi","Palampur","Kullu","Hamirpur","Nahan","Bilaspur"],
    "Jharkhand":              ["Ranchi","Jamshedpur","Dhanbad","Bokaro Steel City","Deoghar","Hazaribagh","Giridih","Ramgarh","Phusro","Medininagar"],
    "Karnataka":              ["Bengaluru","Mysuru","Mangaluru","Hubli","Belagavi","Davangere","Ballari","Vijayapura","Shivamogga","Tumkur","Bidar","Raichur","Udupi","Hassan","Kolar","Mandya","Chitradurga","Gadag"],
    "Kerala":                 ["Thiruvananthapuram","Kochi","Kozhikode","Kollam","Thrissur","Palakkad","Alappuzha","Malappuram","Kannur","Kasaragod","Kottayam","Pathanamthitta","Idukki","Wayanad"],
    "Madhya Pradesh":         ["Indore","Bhopal","Jabalpur","Gwalior","Ujjain","Sagar","Satna","Rewa","Ratlam","Dewas","Murwara","Chhindwara","Shivpuri","Vidisha","Hoshangabad","Burhanpur","Khandwa"],
    "Maharashtra":            ["Mumbai","Pune","Nagpur","Nashik","Aurangabad","Solapur","Amravati","Kolhapur","Nanded","Sangli","Malegaon","Jalgaon","Akola","Latur","Dhule","Ahmednagar","Chandrapur","Parbhani","Thane","Navi Mumbai","Vasai-Virar"],
    "Manipur":                ["Imphal","Thoubal","Kakching","Senapati","Churachandpur"],
    "Meghalaya":              ["Shillong","Tura","Nongstoin","Jowai","Baghmara"],
    "Mizoram":                ["Aizawl","Lunglei","Champhai","Serchhip","Kolasib"],
    "Nagaland":               ["Kohima","Dimapur","Mokokchung","Tuensang","Wokha"],
    "Odisha":                 ["Bhubaneswar","Cuttack","Rourkela","Brahmapur","Sambalpur","Puri","Balasore","Bhadrak","Baripada","Jharsuguda","Angul","Dhenkanal","Balangir","Rayagada","Kendujhar"],
    "Punjab":                 ["Ludhiana","Amritsar","Jalandhar","Patiala","Bathinda","Mohali","Firozpur","Hoshiarpur","Batala","Pathankot","Moga","Abohar","Malerkotla","Khanna","Muktsar"],
    "Rajasthan":              ["Jaipur","Jodhpur","Kota","Bikaner","Ajmer","Udaipur","Bhilwara","Alwar","Bharatpur","Sikar","Pali","Jhunjhunu","Sri Ganganagar","Nagaur","Barmer","Tonk","Beawar","Hanumangarh"],
    "Sikkim":                 ["Gangtok","Namchi","Gyalshing","Mangan"],
    "Tamil Nadu":             ["Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli","Tiruppur","Ranipet","Vellore","Erode","Thoothukkudi","Dindigul","Thanjavur","Kanchipuram","Cuddalore","Kumbakonam","Hosur","Karur","Nagercoil"],
    "Telangana":              ["Hyderabad","Warangal","Nizamabad","Karimnagar","Ramagundam","Khammam","Mahbubnagar","Nalgonda","Adilabad","Suryapet","Miryalaguda"],
    "Tripura":                ["Agartala","Udaipur","Dharmanagar","Kailasahar","Belonia"],
    "Uttar Pradesh":          ["Lucknow","Kanpur","Agra","Varanasi","Meerut","Allahabad","Bareilly","Aligarh","Moradabad","Saharanpur","Gorakhpur","Noida","Firozabad","Loni","Jhansi","Muzaffarnagar","Mathura","Ghaziabad","Shahjahanpur","Rampur","Hapur","Etawah","Faizabad","Bulandshahr","Sambhal","Amroha","Hardoi","Fatehpur","Raebareli","Orai"],
    "Uttarakhand":            ["Dehradun","Haridwar","Roorkee","Haldwani","Rudrapur","Kashipur","Rishikesh","Pithoragarh","Ramnagar","Sitarganj"],
    "West Bengal":            ["Kolkata","Asansol","Siliguri","Durgapur","Bardhaman","Malda","Baharampur","Habra","Kharagpur","Shantipur","Dankuni","Dhulian","Ranaghat","Uluberia","Kalna","Haldia","Raiganj"],
    # Union Territories
    "Delhi":                  ["New Delhi","Delhi","Dwarka","Rohini","Janakpuri","Laxmi Nagar","Shahdara","Pitampura","Vasant Kunj","Saket"],
    "Chandigarh":             ["Chandigarh"],
    "Puducherry":             ["Puducherry","Karaikal","Mahe","Yanam"],
    "Jammu & Kashmir":        ["Srinagar","Jammu","Anantnag","Sopore","Baramulla","Udhampur","Kathua","Poonch"],
    "Ladakh":                 ["Leh","Kargil"],
    "Andaman & Nicobar":      ["Port Blair"],
    "Dadra & Nagar Haveli":   ["Silvassa"],
    "Daman & Diu":            ["Daman","Diu"],
    "Lakshadweep":            ["Kavaratti"],
}

# ─────────────────────────────────────────────────────────────────────────────
# World cities — top cities per country (alphabetical by country)
# ─────────────────────────────────────────────────────────────────────────────
WORLD_CITIES: dict[str, list[str]] = {
    "Afghanistan":        ["Kabul","Kandahar","Herat","Mazar-i-Sharif"],
    "Albania":            ["Tirana","Durrës","Vlorë","Shkodër"],
    "Algeria":            ["Algiers","Oran","Constantine","Annaba","Batna"],
    "Argentina":          ["Buenos Aires","Córdoba","Rosario","Mendoza","La Plata","Mar del Plata","Tucumán","Salta"],
    "Australia":          ["Sydney","Melbourne","Brisbane","Perth","Adelaide","Canberra","Darwin","Hobart"],
    "Austria":            ["Vienna","Graz","Linz","Salzburg","Innsbruck"],
    "Azerbaijan":         ["Baku","Ganja","Sumqayit"],
    "Bangladesh":         ["Dhaka","Chittagong","Khulna","Sylhet","Rajshahi","Comilla"],
    "Belarus":            ["Minsk","Gomel","Mogilev","Vitebsk","Grodno"],
    "Belgium":            ["Brussels","Antwerp","Ghent","Charleroi","Liège","Bruges"],
    "Bolivia":            ["Santa Cruz","La Paz","Cochabamba","Sucre","Oruro"],
    "Bosnia and Herzegovina": ["Sarajevo","Banja Luka","Mostar","Zenica"],
    "Brazil":             ["São Paulo","Rio de Janeiro","Brasília","Salvador","Fortaleza","Belo Horizonte","Manaus","Curitiba","Recife","Porto Alegre","Belém","Goiânia"],
    "Bulgaria":           ["Sofia","Plovdiv","Varna","Burgas","Stara Zagora"],
    "Cambodia":           ["Phnom Penh","Siem Reap","Battambang"],
    "Canada":             ["Toronto","Montreal","Vancouver","Calgary","Edmonton","Ottawa","Winnipeg","Quebec City","Hamilton","Kitchener"],
    "Chile":              ["Santiago","Valparaíso","Concepción","Antofagasta","Viña del Mar","Temuco"],
    "China":              ["Beijing","Shanghai","Guangzhou","Shenzhen","Chengdu","Tianjin","Wuhan","Xi'an","Hangzhou","Nanjing","Chongqing","Shenyang","Dongguan","Harbin","Foshan","Zhengzhou","Qingdao","Kunming","Changsha","Jinan"],
    "Colombia":           ["Bogotá","Medellín","Cali","Barranquilla","Cartagena","Cúcuta","Bucaramanga","Pereira"],
    "Croatia":            ["Zagreb","Split","Rijeka","Osijek"],
    "Cuba":               ["Havana","Santiago de Cuba","Holguín","Camagüey"],
    "Czech Republic":     ["Prague","Brno","Ostrava","Plzeň","Liberec"],
    "Denmark":            ["Copenhagen","Aarhus","Odense","Aalborg","Frederiksberg"],
    "Ecuador":            ["Guayaquil","Quito","Cuenca","Santo Domingo"],
    "Egypt":              ["Cairo","Alexandria","Giza","Shubra El Kheima","Port Said","Suez","Luxor","Mansoura","Aswan"],
    "Ethiopia":           ["Addis Ababa","Dire Dawa","Mek'ele","Adama","Gondar"],
    "Finland":            ["Helsinki","Espoo","Tampere","Vantaa","Oulu","Turku"],
    "France":             ["Paris","Marseille","Lyon","Toulouse","Nice","Nantes","Strasbourg","Bordeaux","Montpellier","Rennes"],
    "Germany":            ["Berlin","Hamburg","Munich","Cologne","Frankfurt","Stuttgart","Düsseldorf","Leipzig","Dortmund","Essen","Bremen","Dresden","Hanover"],
    "Ghana":              ["Accra","Kumasi","Tamale","Sekondi-Takoradi"],
    "Greece":             ["Athens","Thessaloniki","Patras","Heraklion","Larissa"],
    "Hungary":            ["Budapest","Debrecen","Miskolc","Szeged","Pécs"],
    "India":              list({city for cities in INDIA_CITIES.values() for city in cities}),
    "Indonesia":          ["Jakarta","Surabaya","Bandung","Medan","Bekasi","Tangerang","Depok","Semarang","Palembang","Makassar","Bogor","Batam","Pekanbaru"],
    "Iran":               ["Tehran","Mashhad","Isfahan","Karaj","Shiraz","Tabriz","Qom","Ahvaz","Kermanshah"],
    "Iraq":               ["Baghdad","Basra","Mosul","Erbil","Najaf","Karbala"],
    "Ireland":            ["Dublin","Cork","Limerick","Galway","Waterford"],
    "Israel":             ["Jerusalem","Tel Aviv","Haifa","Rishon LeZion","Petah Tikva","Ashdod","Netanya","Beer Sheva"],
    "Italy":              ["Rome","Milan","Naples","Turin","Palermo","Genoa","Bologna","Florence","Bari","Catania","Venice"],
    "Japan":              ["Tokyo","Yokohama","Osaka","Nagoya","Sapporo","Fukuoka","Kobe","Kyoto","Kawasaki","Saitama","Hiroshima","Sendai"],
    "Jordan":             ["Amman","Zarqa","Irbid","Aqaba"],
    "Kazakhstan":         ["Almaty","Nur-Sultan","Shymkent","Karaganda","Aktobe"],
    "Kenya":              ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret"],
    "Kuwait":             ["Kuwait City","Al Ahmadi","Hawalli","Salmiya"],
    "Lebanon":            ["Beirut","Tripoli","Sidon","Tyre"],
    "Libya":              ["Tripoli","Benghazi","Misrata","Bayda"],
    "Malaysia":           ["Kuala Lumpur","George Town","Ipoh","Shah Alam","Petaling Jaya","Johor Bahru","Kuching","Kota Kinabalu","Subang Jaya"],
    "Mexico":             ["Mexico City","Guadalajara","Monterrey","Puebla","Tijuana","León","Juárez","Zapopan","Mérida","San Luis Potosí","Querétaro","Aguascalientes"],
    "Morocco":            ["Casablanca","Rabat","Fes","Marrakesh","Agadir","Tangier","Meknes","Oujda"],
    "Myanmar":            ["Yangon","Mandalay","Naypyidaw","Mawlamyine","Bago"],
    "Nepal":              ["Kathmandu","Pokhara","Lalitpur","Biratnagar","Birgunj"],
    "Netherlands":        ["Amsterdam","Rotterdam","The Hague","Utrecht","Eindhoven","Tilburg","Groningen","Almere"],
    "New Zealand":        ["Auckland","Wellington","Christchurch","Hamilton","Tauranga","Dunedin"],
    "Nigeria":            ["Lagos","Kano","Ibadan","Abuja","Port Harcourt","Benin City","Maiduguri","Zaria","Aba","Jos","Enugu","Onitsha"],
    "Norway":             ["Oslo","Bergen","Trondheim","Stavanger","Tromsø"],
    "Pakistan":           ["Karachi","Lahore","Faisalabad","Rawalpindi","Gujranwala","Peshawar","Multan","Hyderabad","Islamabad","Quetta","Bahawalpur","Sargodha"],
    "Peru":               ["Lima","Arequipa","Trujillo","Chiclayo","Iquitos","Piura","Cusco"],
    "Philippines":        ["Manila","Quezon City","Davao","Cebu","Zamboanga","Antipolo","Pasig","Taguig","Valenzuela"],
    "Poland":             ["Warsaw","Kraków","Łódź","Wrocław","Poznań","Gdańsk","Szczecin","Bydgoszcz","Lublin","Katowice"],
    "Portugal":           ["Lisbon","Porto","Braga","Amadora","Setúbal","Coimbra"],
    "Qatar":              ["Doha","Al Wakrah","Al Khor","Al Rayyan"],
    "Romania":            ["Bucharest","Cluj-Napoca","Timișoara","Iași","Constanța","Craiova","Brașov","Galați"],
    "Russia":             ["Moscow","Saint Petersburg","Novosibirsk","Yekaterinburg","Nizhny Novgorod","Kazan","Chelyabinsk","Omsk","Samara","Rostov-on-Don","Ufa","Krasnoyarsk","Voronezh","Perm","Volgograd"],
    "Saudi Arabia":       ["Riyadh","Jeddah","Mecca","Medina","Dammam","Tabuk","Taif","Abha","Buraidah","Khobar"],
    "Serbia":             ["Belgrade","Novi Sad","Niš","Kragujevac","Subotica"],
    "Singapore":          ["Singapore"],
    "South Africa":       ["Johannesburg","Cape Town","Durban","Pretoria","Port Elizabeth","Bloemfontein","Soweto","Pietermaritzburg"],
    "South Korea":        ["Seoul","Busan","Incheon","Daegu","Daejeon","Gwangju","Suwon","Ulsan","Changwon","Seongnam"],
    "Spain":              ["Madrid","Barcelona","Valencia","Seville","Zaragoza","Málaga","Murcia","Palma","Bilbao","Alicante","Valladolid","Córdoba"],
    "Sri Lanka":          ["Colombo","Kandy","Galle","Jaffna","Negombo","Batticaloa"],
    "Sudan":              ["Khartoum","Omdurman","Port Sudan","Kassala"],
    "Sweden":             ["Stockholm","Gothenburg","Malmö","Uppsala","Västerås","Örebro","Linköping"],
    "Switzerland":        ["Zurich","Geneva","Basel","Bern","Lausanne","Winterthur"],
    "Syria":              ["Damascus","Aleppo","Homs","Latakia","Hama"],
    "Taiwan":             ["Taipei","Kaohsiung","Taichung","Tainan","Hsinchu","Keelung"],
    "Tanzania":           ["Dar es Salaam","Mwanza","Arusha","Dodoma","Mbeya"],
    "Thailand":           ["Bangkok","Nonthaburi","Pak Kret","Hat Yai","Chiang Mai","Udon Thani","Surat Thani","Pattaya","Khon Kaen"],
    "Tunisia":            ["Tunis","Sfax","Sousse","Ettadhamen","Kairouan","Bizerte"],
    "Turkey":             ["Istanbul","Ankara","Izmir","Bursa","Adana","Gaziantep","Konya","Antalya","Kayseri","Mersin","Diyarbakır","Eskişehir"],
    "Uganda":             ["Kampala","Gulu","Lira","Mbarara","Jinja"],
    "Ukraine":            ["Kyiv","Kharkiv","Odessa","Dnipro","Donetsk","Zaporizhzhia","Lviv","Kryvyi Rih","Mykolaiv","Mariupol"],
    "United Arab Emirates": ["Dubai","Abu Dhabi","Sharjah","Al Ain","Ajman","Ras Al Khaimah","Fujairah"],
    "United Kingdom":     ["London","Birmingham","Leeds","Glasgow","Sheffield","Bradford","Edinburgh","Liverpool","Manchester","Bristol","Cardiff","Belfast","Leicester","Coventry","Nottingham"],
    "United States":      ["New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","San Diego","Dallas","Jacksonville","Austin","Fort Worth","Columbus","Charlotte","Indianapolis","San Francisco","Seattle","Denver","Washington DC","Nashville","Oklahoma City","El Paso","Boston","Portland","Las Vegas","Memphis","Louisville","Baltimore","Milwaukee","Albuquerque","Tucson","Fresno","Sacramento","Mesa","Kansas City","Atlanta","Omaha","Colorado Springs","Raleigh","Long Beach","Virginia Beach","Minneapolis","Tampa","New Orleans","Arlington"],
    "Uzbekistan":         ["Tashkent","Samarkand","Namangan","Andijan","Bukhara","Nukus"],
    "Venezuela":          ["Caracas","Maracaibo","Valencia","Barquisimeto","Maracay","Ciudad Guayana"],
    "Vietnam":            ["Ho Chi Minh City","Hanoi","Da Nang","Hai Phong","Can Tho","Bien Hoa","Hue","Nha Trang","Da Lat"],
    "Yemen":              ["Sanaa","Aden","Taiz","Hudaydah","Mukalla"],
    "Zimbabwe":           ["Harare","Bulawayo","Chitungwiza","Mutare","Gweru"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_countries() -> list[str]:
    """Return sorted list of all available countries."""
    return sorted(WORLD_CITIES.keys())


def get_cities_for_country(country: str) -> list[str]:
    """Return sorted list of cities for a given country."""
    return sorted(WORLD_CITIES.get(country, []))


def get_india_states() -> list[str]:
    """Return sorted list of Indian states/UTs."""
    return sorted(INDIA_CITIES.keys())


def get_india_cities_for_state(state: str) -> list[str]:
    """Return sorted cities list for a given Indian state."""
    return sorted(INDIA_CITIES.get(state, []))


def get_all_india_cities() -> list[str]:
    """Return a flat sorted list of all Indian cities."""
    all_cities = {city for cities in INDIA_CITIES.values() for city in cities}
    return sorted(all_cities)


def get_country_for_city(city: str) -> str:
    """Best-effort reverse lookup: return country code for a city name."""
    for country, cities in WORLD_CITIES.items():
        if city in cities:
            return country
    return "India"   # default fallback


# Country → ISO alpha-2 code (used for OWM API calls)
COUNTRY_CODES: dict[str, str] = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Argentina": "AR",
    "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ", "Bangladesh": "BD",
    "Belarus": "BY", "Belgium": "BE", "Bolivia": "BO", "Bosnia and Herzegovina": "BA",
    "Brazil": "BR", "Bulgaria": "BG", "Cambodia": "KH", "Canada": "CA",
    "Chile": "CL", "China": "CN", "Colombia": "CO", "Croatia": "HR", "Cuba": "CU",
    "Czech Republic": "CZ", "Denmark": "DK", "Ecuador": "EC", "Egypt": "EG",
    "Ethiopia": "ET", "Finland": "FI", "France": "FR", "Germany": "DE",
    "Ghana": "GH", "Greece": "GR", "Hungary": "HU", "India": "IN",
    "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ", "Ireland": "IE",
    "Israel": "IL", "Italy": "IT", "Japan": "JP", "Jordan": "JO",
    "Kazakhstan": "KZ", "Kenya": "KE", "Kuwait": "KW", "Lebanon": "LB",
    "Libya": "LY", "Malaysia": "MY", "Mexico": "MX", "Morocco": "MA",
    "Myanmar": "MM", "Nepal": "NP", "Netherlands": "NL", "New Zealand": "NZ",
    "Nigeria": "NG", "Norway": "NO", "Pakistan": "PK", "Peru": "PE",
    "Philippines": "PH", "Poland": "PL", "Portugal": "PT", "Qatar": "QA",
    "Romania": "RO", "Russia": "RU", "Saudi Arabia": "SA", "Serbia": "RS",
    "Singapore": "SG", "South Africa": "ZA", "South Korea": "KR", "Spain": "ES",
    "Sri Lanka": "LK", "Sudan": "SD", "Sweden": "SE", "Switzerland": "CH",
    "Syria": "SY", "Taiwan": "TW", "Tanzania": "TZ", "Thailand": "TH",
    "Tunisia": "TN", "Turkey": "TR", "Uganda": "UG", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States": "US",
    "Uzbekistan": "UZ", "Venezuela": "VE", "Vietnam": "VN", "Yemen": "YE",
    "Zimbabwe": "ZW",
}


def get_country_code(country: str) -> str:
    """Return ISO alpha-2 code for a country name."""
    return COUNTRY_CODES.get(country, "IN")
