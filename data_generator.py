"""
Synthetic House Price Dataset Generator — All India States & UTs
Generates realistic heterogeneous housing data for all 28 states + 8 UTs.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# All 28 States + 8 UTs
ALL_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    # UTs
    'Delhi', 'Chandigarh', 'Puducherry', 'Jammu & Kashmir',
    'Ladakh', 'Lakshadweep', 'Andaman & Nicobar', 'Dadra & Nagar Haveli'
]

# Major real estate areas per state/UT
STATE_AREAS = {
    'Andhra Pradesh':        ['Vijayawada', 'Visakhapatnam', 'Guntur', 'Tirupati', 'Nellore', 'Amaravati', 'Rajahmundry', 'Kurnool'],
    'Arunachal Pradesh':     ['Itanagar', 'Naharlagun', 'Pasighat', 'Tawang', 'Ziro'],
    'Assam':                 ['Guwahati', 'Dibrugarh', 'Silchar', 'Jorhat', 'Dispur', 'Tezpur', 'Nagaon'],
    'Bihar':                 ['Patna', 'Gaya', 'Muzaffarpur', 'Bhagalpur', 'Darbhanga', 'Purnia', 'Nalanda'],
    'Chhattisgarh':          ['Raipur', 'Bhilai', 'Bilaspur', 'Durg', 'Korba', 'Rajnandgaon'],
    'Goa':                   ['Panaji', 'Margao', 'Vasco da Gama', 'Mapusa', 'Ponda', 'Calangute', 'Baga', 'Candolim'],
    'Gujarat':               ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'SG Highway', 'Bopal', 'Gandhinagar', 'Anand'],
    'Haryana':               ['Gurugram', 'Faridabad', 'Panipat', 'Ambala', 'Hisar', 'DLF City', 'Sohna Road', 'Palam Vihar'],
    'Himachal Pradesh':      ['Shimla', 'Manali', 'Dharamshala', 'Solan', 'Kullu', 'Mandi', 'Palampur'],
    'Jharkhand':             ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro', 'Hazaribagh', 'Deoghar'],
    'Karnataka':             ['Whitefield', 'Koramangala', 'Indiranagar', 'Hebbal', 'Mysuru', 'Mangaluru', 'Bellandur', 'Electronic City'],
    'Kerala':                ['Kochi', 'Thiruvananthapuram', 'Kozhikode', 'Thrissur', 'Kakkanad', 'Panampilly Nagar', 'Kollam', 'Palakkad'],
    'Madhya Pradesh':        ['Bhopal', 'Indore', 'Gwalior', 'Jabalpur', 'Ujjain', 'Vijay Nagar', 'Vijay Nagar Indore', 'Rewa'],
    'Maharashtra':           ['Baner', 'Juhu', 'Andheri', 'Thane', 'Pune Camp', 'Navi Mumbai', 'Nagpur', 'Aurangabad'],
    'Manipur':               ['Imphal', 'Thoubal', 'Bishnupur', 'Churachandpur', 'Kakching'],
    'Meghalaya':             ['Shillong', 'Tura', 'Jowai', 'Nongpoh', 'Cherrapunjee'],
    'Mizoram':               ['Aizawl', 'Lunglei', 'Champhai', 'Serchhip', 'Kolasib'],
    'Nagaland':              ['Kohima', 'Dimapur', 'Mokokchung', 'Tuensang', 'Wokha'],
    'Odisha':                ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Puri', 'Sambalpur', 'Berhampur', 'Infocity'],
    'Punjab':                ['Ludhiana', 'Amritsar', 'Jalandhar', 'Mohali', 'Patiala', 'SAS Nagar', 'Zirakpur', 'Bathinda'],
    'Rajasthan':             ['Vaishali Nagar', 'Malviya Nagar', 'C-Scheme', 'Mansarovar', 'Jodhpur', 'Udaipur', 'Ajmer', 'Kota'],
    'Sikkim':                ['Gangtok', 'Namchi', 'Mangan', 'Gyalshing', 'Ravangla'],
    'Tamil Nadu':            ['Anna Nagar', 'T Nagar', 'Velachery', 'OMR', 'Coimbatore RS Puram', 'Madurai', 'Trichy', 'Perambur'],
    'Telangana':             ['Banjara Hills', 'Jubilee Hills', 'Hitech City', 'Kondapur', 'Gachibowli', 'Kukatpally', 'Secunderabad', 'Warangal'],
    'Tripura':               ['Agartala', 'Udaipur Tripura', 'Dharmanagar', 'Kailasahar', 'Belonia'],
    'Uttar Pradesh':         ['Gomti Nagar', 'Hazratganj', 'Noida Sector 44', 'Greater Noida', 'Varanasi', 'Agra Cantt', 'Kanpur', 'Mathura'],
    'Uttarakhand':           ['Dehradun', 'Haridwar', 'Nainital', 'Rishikesh', 'Haldwani', 'Mussoorie', 'Roorkee'],
    'West Bengal':           ['Salt Lake', 'Rajarhat', 'Alipore', 'New Town', 'Howrah', 'Durgapur', 'Siliguri', 'Asansol'],
    'Delhi':                 ['Connaught Place', 'Dwarka', 'Rohini', 'Vasant Kunj', 'Saket', 'Lajpat Nagar', 'Janakpuri', 'Pitampura'],
    'Chandigarh':            ['Sector 17', 'Sector 22', 'Sector 35', 'Sector 44', 'Manimajra', 'Panchkula', 'Mohali Phase 1'],
    'Puducherry':            ['White Town', 'Nellithope', 'Reddiyarpalayam', 'Oulgaret', 'Villianur'],
    'Jammu & Kashmir':       ['Srinagar', 'Jammu', 'Anantnag', 'Baramulla', 'Udhampur', 'Rajouri', 'Sopore'],
    'Ladakh':                ['Leh', 'Kargil', 'Nubra', 'Zanskar', 'Drass'],
    'Lakshadweep':           ['Kavaratti', 'Agatti', 'Amini', 'Andrott', 'Minicoy'],
    'Andaman & Nicobar':     ['Port Blair', 'Car Nicobar', 'Diglipur', 'Havelock Island', 'Neil Island'],
    'Dadra & Nagar Haveli':  ['Silvassa', 'Amli', 'Khanvel', 'Naroli', 'Rakholi'],
}

# Base price in INR for each state
STATE_BASE_PRICE = {
    'Andhra Pradesh': 5500000,   'Arunachal Pradesh': 2500000, 'Assam': 3000000,
    'Bihar': 2800000,            'Chhattisgarh': 3200000,      'Goa': 8500000,
    'Gujarat': 5500000,          'Haryana': 8000000,           'Himachal Pradesh': 4500000,
    'Jharkhand': 2900000,        'Karnataka': 7200000,         'Kerala': 6000000,
    'Madhya Pradesh': 3500000,   'Maharashtra': 9500000,       'Manipur': 2200000,
    'Meghalaya': 2800000,        'Mizoram': 2300000,           'Nagaland': 2100000,
    'Odisha': 3300000,           'Punjab': 5000000,            'Rajasthan': 4200000,
    'Sikkim': 3500000,           'Tamil Nadu': 6800000,        'Telangana': 7500000,
    'Tripura': 2400000,          'Uttar Pradesh': 4500000,     'Uttarakhand': 4800000,
    'West Bengal': 5800000,      'Delhi': 12000000,            'Chandigarh': 7000000,
    'Puducherry': 4800000,       'Jammu & Kashmir': 4000000,   'Ladakh': 3500000,
    'Lakshadweep': 3000000,      'Andaman & Nicobar': 4500000, 'Dadra & Nagar Haveli': 3800000,
}

PROPERTY_TYPES = ['Apartment', 'Independent House', 'Villa', 'Builder Floor', 'Penthouse']
CONDITIONS     = ['Excellent', 'Good', 'Fair', 'Poor']
GARAGE_TYPES   = ['Covered Parking', 'Open Parking', 'None', 'Basement Parking']
SCHOOL_RATINGS = ['A+', 'A', 'B+', 'B', 'C+', 'C']

CONDITION_MULT = {'Excellent': 1.20, 'Good': 1.05, 'Fair': 0.90, 'Poor': 0.75}
SCHOOL_MULT    = {'A+': 1.15, 'A': 1.10, 'B+': 1.05, 'B': 1.00, 'C+': 0.95, 'C': 0.90}


def generate_dataset(n_samples=5000):
    state_arr    = np.random.choice(ALL_STATES, n_samples)
    area_arr     = np.array([np.random.choice(STATE_AREAS[s]) for s in state_arr])
    prop_arr     = np.random.choice(PROPERTY_TYPES, n_samples, p=[0.40, 0.25, 0.15, 0.15, 0.05])
    cond_arr     = np.random.choice(CONDITIONS,     n_samples, p=[0.25, 0.45, 0.20, 0.10])
    garage_arr   = np.random.choice(GARAGE_TYPES,   n_samples, p=[0.40, 0.25, 0.25, 0.10])
    school_arr   = np.random.choice(SCHOOL_RATINGS, n_samples, p=[0.10, 0.20, 0.20, 0.25, 0.15, 0.10])

    sqft         = np.random.normal(1400, 600, n_samples).clip(400, 5000).astype(int)
    bedrooms     = np.random.choice([1,2,3,4,5,6], n_samples, p=[0.08,0.20,0.38,0.24,0.07,0.03])
    bathrooms    = np.clip(bedrooms * np.random.uniform(0.5,1.0,n_samples), 1, 5).astype(int)
    year_built   = np.random.randint(1975, 2024, n_samples)
    lot_size     = np.random.normal(2000, 800, n_samples).clip(500, 12000).astype(int)
    floors       = np.random.choice([1,2,3,4,5], n_samples, p=[0.20,0.30,0.25,0.15,0.10])
    has_pool     = np.random.choice([0,1], n_samples, p=[0.80,0.20])
    has_basement = np.random.choice([0,1], n_samples, p=[0.65,0.35])
    distance_cbd = np.random.uniform(0.5, 40, n_samples).round(1)
    crime_rate   = np.random.uniform(1, 10, n_samples).round(2)
    walk_score   = np.random.randint(20, 100, n_samples)
    renovation_year = np.where(
        np.random.random(n_samples) > 0.4,
        np.random.randint(2000, 2024, n_samples),
        year_built
    )

    prices = []
    for i in range(n_samples):
        base  = STATE_BASE_PRICE[state_arr[i]]
        price = base
        price += sqft[i]        * np.random.uniform(3000, 8000)
        price += bedrooms[i]    * np.random.uniform(200000, 500000)
        price += bathrooms[i]   * np.random.uniform(100000, 300000)
        price *= CONDITION_MULT[cond_arr[i]]
        price *= SCHOOL_MULT[school_arr[i]]
        if has_pool[i]:     price += np.random.uniform(500000, 2000000)
        if has_basement[i]: price += np.random.uniform(300000, 800000)
        price -= distance_cbd[i]  * np.random.uniform(50000, 200000)
        price -= crime_rate[i]    * np.random.uniform(100000, 300000)
        price += walk_score[i]    * np.random.uniform(5000, 20000)
        price += (2024 - year_built[i]) * np.random.uniform(-10000, -3000)
        if garage_arr[i] != 'None': price += np.random.uniform(200000, 600000)
        price *= np.random.uniform(0.92, 1.08)
        prices.append(max(800000, price))

    df = pd.DataFrame({
        'city': state_arr, 'neighborhood': area_arr,
        'property_type': prop_arr, 'condition': cond_arr,
        'garage_type': garage_arr, 'school_rating': school_arr,
        'sqft': sqft, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'year_built': year_built, 'lot_size': lot_size, 'floors': floors,
        'has_pool': has_pool, 'has_basement': has_basement,
        'distance_cbd': distance_cbd, 'crime_rate': crime_rate,
        'walk_score': walk_score, 'renovation_year': renovation_year,
        'price': np.array(prices).astype(int)
    })
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/house_data.csv', index=False)
    print(f"[DataGenerator] Dataset created: {n_samples} samples -> data/house_data.csv")
    return df


if __name__ == '__main__':
    df = generate_dataset()
    print(df.describe())
