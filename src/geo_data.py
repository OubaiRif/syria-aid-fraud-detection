"""
Geography model for the NW-Syria cross-border program.

Real governorate/district (admin1/admin2) names give the dataset
credibility; zone_x/zone_y are synthetic coordinates within a 0-100 grid
per district, which keeps distance math simple (Euclidean) while avoiding
the need for real GPS data.
"""

DISTRICTS = {
    "Idlib": [
        "Idlib",
        "Harim",
        "Ariha",
        "Jisr-Ash-Shugur",
        "Maarrat An Nu'man",
    ],
    "Aleppo": [
        "A'zaz",
        "Afrin",
        "Al Bab",
        "Jarablus",
        "Mare'",
    ],
}

# Flat list of (governorate, district) pairs, and district -> governorate lookup
GOVERNORATE_DISTRICT_PAIRS = [
    (gov, dist) for gov, dists in DISTRICTS.items() for dist in dists
]
DISTRICT_TO_GOVERNORATE = {dist: gov for gov, dists in DISTRICTS.items() for dist in dists}
ALL_DISTRICTS = [dist for _, dist in GOVERNORATE_DISTRICT_PAIRS]


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
