# config.py

WIDTH, HEIGHT = 1100, 720
FPS = 60

CITY_MARGIN = 60

# Robot settings
NUM_ROBOTS = 10
ROBOT_RADIUS = 10
PATROL_SPEED = 2.1
RESPONSE_SPEED = 3.2

# Hazard timing (seconds)
HAZARD_COOLDOWN_MIN = 5
HAZARD_COOLDOWN_MAX = 10

# Sensing & communication
SENSE_RADIUS = 140.0      # robots within this radius of building detect hazard
COMM_RADIUS = 220.0       # robots within this distance can exchange info

# Consensus
CONSENSUS_THRESHOLD = 0.7  # belief confidence to "commit" to hazard building

# Arrival / aggregation
ARRIVAL_RADIUS_FACTOR = 1.5  # * ROBOT_RADIUS

# Metrics
METRICS_CSV = "metrics_log.csv"
