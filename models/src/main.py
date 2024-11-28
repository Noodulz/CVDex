import utils as u
import model as m

if __name__=="__main__":
    resolution = 224
    dataset = u.data_loader("../dataset/dataset/", resolution, resolution)
    labels = u.get_labels(dataset)
    (training, validation, testing) = u.dataset_distribution(dataset)
    model = m.VisionModel(len(labels), resolution)
    epochs = 200

    m.train_model(model, epochs, training, validation)
    m.evaluate(model, testing)
    print("Hello world!")
