from datasets.load_data.tourism import TourismDataset
from datasets.load_data.gluonts import GluontsDataset

DATASETS = {
    'M3': M3Dataset,
    'Tourism': TourismDataset,
    'Gluonts': GluontsDataset,
}

DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]
