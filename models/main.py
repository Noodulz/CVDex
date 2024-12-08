import src.utils as u
import src.model as m

if __name__=="__main__":
    resolution = 224
    transform = u.transform_autocomp(resolution, resolution)
    dataset = u.data_loader("./dataset/dataset/", transform)
    labels = u.get_labels(dataset)
    (training, validation, testing) = u.stratified_distribution(dataset)
    model = m.VisionModel(len(labels), resolution)
    epochs = 10

    m.train_model(model, epochs, training, validation)
    m.evaluate(model, testing)
    print("Hello world!")
