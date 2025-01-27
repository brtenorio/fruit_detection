if __name__=="__main__":
    """
    Run this script if you want to retrain the model and persist a new model.keras file
    """
    from fruit_detection.data_generator import * 
    from fruit_detection.train_model import train_model
    from fruit_detection.get_model import get_model
    from fruit_detection.evaluate_model import evaluate_model
    from fruit_detection.save_model import save_model

    model = get_model()
    model = train_model(model)
    save_model(model)
    eval = evaluate_model(test_generator)
    print('Test loss: ', eval[0])
    print('Test Accuracy: ', eval[1])