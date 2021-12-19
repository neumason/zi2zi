from keras.models import load_model
from keras.utils.vis_utils import plot_model
model = load_model("decoder.h5")
plot_model(model, to_file="model.png", show_shapes=True)


