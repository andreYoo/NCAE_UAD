import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def offset_image(x, y, label, bar_is_too_short, ax):
    response = requests.get(f'https://www.countryflags.io/{label}/flat/64.png')
    img = plt.imread(BytesIO(response.content))
    im = OffsetImage(img, zoom=0.65)
    im.image.axes = ax
    x_offset = -25
    if bar_is_too_short:
        x = 0
    ab = AnnotationBbox(im, (x, y), xybox=(x_offset, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)

labels = ['CW', 'CV', 'GW', 'SX', 'DO']
colors = ['crimson', 'dodgerblue', 'teal', 'limegreen', 'gold']
values = 2 ** np.random.randint(2, 10, len(labels))

height = 0.9
plt.barh(y=labels, width=values, height=height, color=colors, align='center', alpha=0.8)

max_value = values.max()
for i, (label, value) in enumerate(zip(labels, values)):
    offset_image(value, i, label, bar_is_too_short=value < max_value / 10, ax=plt.gca())
plt.subplots_adjust(left=0.15)
plt.show()