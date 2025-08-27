import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class GrazingDataGenerator:
    def __init__(self, output_folder='grazing_data'):
        self.output_folder = output_folder
        self.create_output_folder()

        # First generate constraints, then use them in data generation
        self.constraints = self.generate_constraints()

    def create_output_folder(self):
        """Create output folder if it doesn't exist"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def generate_constraints(self):
        """Generate realistic grazing constraints for the region"""
        constraints = {
            "zone_restrictions": {
                "protected_areas": [8, 9],  # Zones 8 and 9 are nature reserves
                "seasonal_closures": {
                    "breeding_season": {
                        "zones": [3, 7],
                        "months": [4, 5, 6],  # April-June breeding season
                        "reason": "Wildlife breeding protection"
                    },
                    "vegetation_recovery": {
                        "zones": [1, 2, 5],
                        "months": [2, 3],  # Feb-March recovery period
                        "reason": "Vegetation recovery period"
                    }
                },
                "weather_based": {
                    "flood_prone": {
                        "zones": [4, 6],
                        "trigger": "rainfall > 15mm",
                        "reason": "Flood risk"
                    },
                    "extreme_temperature": {
                        "zones": [0, 1, 2],  # Higher elevation zones
                        "trigger": "temperature < 5C",
                        "reason": "Cold weather protection"
                    }
                }
            },
            "carrying_capacity_limits": {
                "max_sheep_per_hectare": 8,
                "max_consecutive_days": 7,
                "recovery_period_days": 14
            },
            "water_access_requirements": {
                "max_distance_from_water_km": 2.0,
                "zones_without_water": [8, 9]  # These zones lack water sources
            },
            "terrain_restrictions": {
                "max_slope_degrees": 25,
                "unsafe_terrain_zones": [9]  # Rocky/dangerous terrain
            }
        }

        # Save constraints to JSON file
        with open(f'{self.output_folder}/grazing_constraints.json', 'w') as f:
            json.dump(constraints, f, indent=2)

        return constraints

    def is_zone_accessible(self, zone_id, date, weather_data=None):
        """Check if a zone is accessible based on constraints"""
        month = date.month

        # Check protected areas
        if zone_id in self.constraints["zone_restrictions"]["protected_areas"]:
            return False, "Protected area"

        # Check seasonal closures
        seasonal = self.constraints["zone_restrictions"]["seasonal_closures"]
        for closure_name, closure_data in seasonal.items():
            if zone_id in closure_data["zones"] and month in closure_data["months"]:
                return False, closure_data["reason"]

        # Check weather-based restrictions if weather data provided
        if weather_data:
            weather_restrictions = self.constraints["zone_restrictions"]["weather_based"]

            # Flood check
            if (zone_id in weather_restrictions["flood_prone"]["zones"] and
                weather_data.get("rainfall", 0) > 15):
                return False, weather_restrictions["flood_prone"]["reason"]

            # Temperature check
            if (zone_id in weather_restrictions["extreme_temperature"]["zones"] and
                weather_data.get("temperature", 20) < 5):
                return False, weather_restrictions["extreme_temperature"]["reason"]

        return True, "Accessible"

    def generate_weather_data(self):
        """Generate weather data for the region"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        weather_data = []
        current_date = start_date

        while current_date <= end_date:
            # Seasonal temperature patterns for Ifrane (Morocco)
            day_of_year = current_date.timetuple().tm_yday
            base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

            # Add daily variation and noise
            temperature = base_temp + np.random.normal(0, 3)

            # Humidity varies with season and temperature
            base_humidity = 60 - (temperature - 15) * 0.5
            humidity = np.clip(base_humidity + np.random.normal(0, 10), 20, 90)

            # Rainfall patterns (more in winter/spring)
            if current_date.month in [11, 12, 1, 2, 3, 4]:
                rainfall_prob = 0.3
                avg_rainfall = 8
            else:
                rainfall_prob = 0.15
                avg_rainfall = 3

            if np.random.random() < rainfall_prob:
                rainfall = np.random.exponential(avg_rainfall)
            else:
                rainfall = 0

            weather_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'rainfall': round(rainfall, 1),
                'description': 'measured_data'
            })

            current_date += timedelta(days=1)

        df = pd.DataFrame(weather_data)
        df.to_csv(f'{self.output_folder}/weather_data.csv', index=False)
        return df

    def generate_vegetation_data(self, weather_df):
        """Generate vegetation data considering constraints"""
        start_date = datetime(2024, 1, 1)
        vegetation_data = []

        # Generate weekly data for each zone
        for week in range(52):
            current_date = start_date + timedelta(weeks=week)

            # Get weather data for this week
            week_weather = weather_df[
                pd.to_datetime(weather_df['date']).dt.isocalendar().week == week + 1
            ]
            avg_temp = week_weather['temperature'].mean() if len(week_weather) > 0 else 15
            total_rainfall = week_weather['rainfall'].sum() if len(week_weather) > 0 else 0

            for zone_id in range(1, 11):  # Zones 1-10
                # Check if zone is accessible
                accessible, reason = self.is_zone_accessible(
                    zone_id, current_date,
                    {"temperature": avg_temp, "rainfall": total_rainfall}
                )

                if accessible:
                    # Normal vegetation calculation
                    # Seasonal NDVI pattern
                    month = current_date.month
                    if month in [3, 4, 5]:  # Spring - best grazing
                        base_ndvi = 0.7
                    elif month in [9, 10, 11]:  # Autumn - good grazing
                        base_ndvi = 0.6
                    elif month in [12, 1, 2]:  # Winter - poor grazing
                        base_ndvi = 0.3
                    else:  # Summer - moderate grazing
                        base_ndvi = 0.4

                    # Weather effects on vegetation
                    temp_factor = 1 - abs(avg_temp - 20) / 30
                    rain_factor = min(total_rainfall / 20, 1.0)

                    ndvi = base_ndvi * temp_factor * rain_factor
                    ndvi = np.clip(ndvi + np.random.normal(0, 0.05), 0, 1)

                    # Zone-specific adjustments
                    if zone_id in [8, 9]:  # Higher elevation zones
                        ndvi *= 0.8
                    elif zone_id in [4, 6]:  # Flood-prone areas - better soil
                        ndvi *= 1.1

                    biomass = ndvi * 1200  # kg per hectare
                    carrying_capacity = min(ndvi * 12, self.constraints["carrying_capacity_limits"]["max_sheep_per_hectare"])

                    # Grass quality based on NDVI
                    if ndvi > 0.6:
                        grass_quality = 'excellent'
                    elif ndvi > 0.4:
                        grass_quality = 'good'
                    elif ndvi > 0.2:
                        grass_quality = 'poor'
                    else:
                        grass_quality = 'very_poor'

                else:
                    # Restricted zone - set poor values
                    ndvi = 0.0
                    biomass = 0.0
                    carrying_capacity = 0.0
                    grass_quality = 'restricted'

                vegetation_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'zone_id': zone_id,
                    'ndvi': round(ndvi, 3),
                    'biomass_kg_per_hectare': round(biomass, 1),
                    'carrying_capacity_sheep_per_hectare': round(carrying_capacity, 1),
                    'grass_quality': grass_quality,
                    'accessible': accessible,
                    'restriction_reason': reason if not accessible else None
                })

        df = pd.DataFrame(vegetation_data)
        df.to_csv(f'{self.output_folder}/vegetation_data.csv', index=False)
        return df

    def generate_livestock_tracking(self, vegetation_df, weather_df):
        """Generate livestock tracking data respecting constraints"""
        livestock_data = []
        start_date = datetime(2024, 1, 1)

        # Track 3 herds
        for herd_id in range(1, 4):
            current_zone = np.random.choice([1, 2, 3])  # Start in accessible zones
            days_in_zone = 0

            for day in range(365):
                current_date = start_date + timedelta(days=day)

                # Get weather for this day
                day_weather = weather_df[weather_df['date'] == current_date.strftime('%Y-%m-%d')]
                weather_data = day_weather.iloc[0].to_dict() if len(day_weather) > 0 else {}

                # Get vegetation data for current zone
                veg_data = vegetation_df[
                    (vegetation_df['zone_id'] == current_zone) &
                    (pd.to_datetime(vegetation_df['date']) <= current_date)
                ].tail(1)

                accessible = True
                if len(veg_data) > 0:
                    accessible = veg_data.iloc[0]['accessible']

                # Force move if zone becomes inaccessible or overgrazed
                max_days = self.constraints["carrying_capacity_limits"]["max_consecutive_days"]
                if not accessible or days_in_zone >= max_days:
                    # Find accessible zone
                    possible_zones = []
                    for test_zone in range(1, 11):
                        zone_accessible, _ = self.is_zone_accessible(test_zone, current_date, weather_data)
                        if zone_accessible:
                            possible_zones.append(test_zone)

                    if possible_zones:
                        current_zone = np.random.choice(possible_zones)
                        days_in_zone = 1
                    else:
                        # Emergency: stay in current zone but record violation
                        days_in_zone += 1
                else:
                    days_in_zone += 1

                # Calculate grazing hours based on season and weather
                base_hours = 8
                if weather_data.get('temperature', 15) < 5:
                    grazing_hours = base_hours * 0.6
                elif weather_data.get('rainfall', 0) > 10:
                    grazing_hours = base_hours * 0.4
                else:
                    grazing_hours = base_hours

                # Distance traveled (more if forced to move due to constraints)
                if days_in_zone == 1:
                    distance = np.random.uniform(3, 8)  # Moving to new zone
                else:
                    distance = np.random.uniform(0.5, 3)  # Normal grazing

                # Risk factors (higher in restricted/marginal zones)
                if not accessible:
                    animals_lost = np.random.poisson(0.5)
                    predator_encounters = np.random.poisson(0.3)
                else:
                    animals_lost = np.random.poisson(0.05)
                    predator_encounters = np.random.poisson(0.1)

                livestock_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'herd_id': herd_id,
                    'current_zone': current_zone,
                    'grazing_hours': round(grazing_hours, 1),
                    'distance_traveled_km': round(distance, 1),
                    'animals_lost': animals_lost,
                    'predator_encounters': predator_encounters,
                    'days_in_zone': days_in_zone,
                    'zone_accessible': accessible,
                    'constraint_violation': not accessible
                })

        df = pd.DataFrame(livestock_data)
        df.to_csv(f'{self.output_folder}/livestock_tracking.csv', index=False)
        return df

    def generate_topographical_data(self):
        """Generate topographical data with terrain constraints"""
        topo_data = []

        # Generate grid points across the region
        lat_range = np.linspace(33.5, 33.6, 50)  # Ifrane region
        lon_range = np.linspace(-5.2, -5.0, 50)

        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                # Determine which zone this point belongs to
                zone_id = ((i // 5) % 10) + 1

                # Base elevation (Ifrane is mountainous)
                base_elevation = 1650 + np.random.normal(0, 200)

                # Zone-specific elevation adjustments
                if zone_id in [8, 9]:  # Higher, more dangerous zones
                    elevation = base_elevation + 300
                    slope = np.random.uniform(15, 35)
                    terrain_type = 'mountainous'
                elif zone_id in [4, 6]:  # Lower, flatter zones
                    elevation = base_elevation - 100
                    slope = np.random.uniform(2, 12)
                    terrain_type = 'valley'
                else:
                    elevation = base_elevation
                    slope = np.random.uniform(5, 20)
                    terrain_type = 'hills'

                # Apply terrain restrictions
                slope_limit = self.constraints["terrain_restrictions"]["max_slope_degrees"]
                terrain_suitable = slope <= slope_limit

                if zone_id in self.constraints["terrain_restrictions"]["unsafe_terrain_zones"]:
                    terrain_suitable = False
                    slope = np.random.uniform(25, 45)  # Make it clearly unsuitable

                topo_data.append({
                    'latitude': round(lat, 6),
                    'longitude': round(lon, 6),
                    'elevation_m': round(elevation, 1),
                    'slope_degrees': round(slope, 1),
                    'terrain_type': terrain_type,
                    'zone_id': zone_id,
                    'suitable_for_grazing': terrain_suitable
                })

        df = pd.DataFrame(topo_data)
        df.to_csv(f'{self.output_folder}/topographical_data.csv', index=False)
        return df

    def generate_water_sources(self):
        """Generate water sources considering access constraints"""
        water_sources = [
            {'name': 'Ain Tit', 'type': 'spring', 'lat': 33.525, 'lon': -5.15, 'seasonal': False, 'zone': 1},
            {'name': 'Oued Tizguit', 'type': 'stream', 'lat': 33.535, 'lon': -5.12, 'seasonal': True, 'zone': 2},
            {'name': 'Source Atlas', 'type': 'spring', 'lat': 33.545, 'lon': -5.08, 'seasonal': False, 'zone': 3},
            {'name': 'Bassin Collecteur', 'type': 'pond', 'lat': 33.540, 'lon': -5.16, 'seasonal': False, 'zone': 4},
            {'name': 'Puits Pastoral', 'type': 'well', 'lat': 33.530, 'lon': -5.11, 'seasonal': False, 'zone': 5},
            {'name': 'Ruisseau Saisonnier', 'type': 'stream', 'lat': 33.555, 'lon': -5.14, 'seasonal': True, 'zone': 6},
            {'name': 'Source Montagne', 'type': 'spring', 'lat': 33.520, 'lon': -5.09, 'seasonal': False, 'zone': 7},
            # Note: Zones 8 and 9 have no water sources (constraint)
            {'name': 'Mare Temporaire', 'type': 'pond', 'lat': 33.565, 'lon': -5.13, 'seasonal': True, 'zone': 10}
        ]

        # Add water access information based on constraints
        for source in water_sources:
            source['accessible'] = source['zone'] not in self.constraints["water_access_requirements"]["zones_without_water"]

        df = pd.DataFrame(water_sources)
        df.to_csv(f'{self.output_folder}/water_sources.csv', index=False)
        return df

    def generate_satellite_info(self):
        """Generate satellite information"""
        satellite_info = {
            "data_sources": {
                "vegetation": "Sentinel-2",
                "weather": "MODIS Terra/Aqua",
                "topography": "SRTM DEM"
            },
            "update_frequency": {
                "vegetation": "weekly",
                "weather": "daily",
                "topography": "static"
            },
            "spatial_resolution": {
                "vegetation": "10m",
                "weather": "1km",
                "topography": "30m"
            },
            "region": {
                "name": "Ifrane Grazing Area",
                "center_lat": 33.533,
                "center_lon": -5.11,
                "area_km2": 25
            }
        }

        with open(f'{self.output_folder}/satellite_info.json', 'w') as f:
            json.dump(satellite_info, f, indent=2)

        return satellite_info

    def generate_all_data(self):
        """Generate complete dataset with constraints"""
        print("Generating comprehensive grazing dataset with constraints...")
        print("=" * 60)

        # Generate constraints first
        print("✓ Grazing constraints defined")

        # Generate weather data (base for other calculations)
        print("Generating weather data...")
        weather_df = self.generate_weather_data()
        print(f"✓ Weather data: {len(weather_df)} records")

        # Generate vegetation data (uses weather and constraints)
        print("Generating vegetation data with constraint integration...")
        vegetation_df = self.generate_vegetation_data(weather_df)
        print(f"✓ Vegetation data: {len(vegetation_df)} records")

        # Generate topographical data (includes terrain constraints)
        print("Generating topographical data...")
        topo_df = self.generate_topographical_data()
        print(f"✓ Topographical data: {len(topo_df)} records")

        # Generate water sources (considers water access constraints)
        print("Generating water sources data...")
        water_df = self.generate_water_sources()
        print(f"✓ Water sources: {len(water_df)} records")

        # Generate livestock tracking (respects all constraints)
        print("Generating livestock tracking with constraint compliance...")
        livestock_df = self.generate_livestock_tracking(vegetation_df, weather_df)
        print(f"✓ Livestock data: {len(livestock_df)} records")

        # Generate satellite info
        print("Generating satellite metadata...")
        satellite_info = self.generate_satellite_info()
        print("✓ Satellite info created")

        print("\n" + "=" * 60)
        print("DATA GENERATION COMPLETE WITH CONSTRAINTS INTEGRATION!")
        print("=" * 60)

        # Print constraint summary
        print("\nCONSTRAINT SUMMARY:")
        print(f"- Protected areas: Zones {self.constraints['zone_restrictions']['protected_areas']}")
        print(f"- Seasonal closures: {len(self.constraints['zone_restrictions']['seasonal_closures'])} types")
        print(f"- Weather-based restrictions: {len(self.constraints['zone_restrictions']['weather_based'])} types")
        print(f"- Max carrying capacity: {self.constraints['carrying_capacity_limits']['max_sheep_per_hectare']} sheep/hectare")
        print(f"- Max consecutive days: {self.constraints['carrying_capacity_limits']['max_consecutive_days']} days")

        # Analyze constraint violations in the data
        violations = livestock_df['constraint_violation'].sum()
        total_records = len(livestock_df)
        print(f"\nCONSTRAINT VIOLATIONS IN DATA:")
        print(f"- Total violations: {violations}/{total_records} ({violations/total_records*100:.1f}%)")

        restricted_zones = vegetation_df[vegetation_df['accessible'] == False]['zone_id'].unique()
        print(f"- Zones with restrictions: {sorted(restricted_zones)}")

        return {
            'weather': weather_df,
            'vegetation': vegetation_df,
            'topography': topo_df,
            'water_sources': water_df,
            'livestock': livestock_df,
            'satellite_info': satellite_info,
            'constraints': self.constraints
        }

# Usage
if __name__ == "__main__":
    # Create generator and generate all data
    generator = GrazingDataGenerator('grazing_data')
    data = generator.generate_all_data()

    print("\n🎯 Data generation complete!")
    print("📁 Check 'grazing_data' folder for all generated files")
    print("🔒 Constraints are now integrated into all datasets")
    print("📊 Ready for constraint-aware AI training!")