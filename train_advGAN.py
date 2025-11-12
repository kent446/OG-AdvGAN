
from ultralytics.models.yolo.detect.train import DetectionTrainer
from advGAN import AdvGAN
from ultralytics import YOLO
import utils


if __name__ == "__main__":
    training_parameters = {
        "EPOCHS": 60,
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 0.0001
    }
    targeted_model_file_name = 'best.pt'
    model_num_labels = 21
    image_nc = 3
    #图片通道数
    BOX_MIN = 0
    BOX_MAX = 1
    model_path = './models/'
    grad_path = './grads_all/'
    # Define what device we are using
    device = utils.define_device()


    V8 = YOLO(targeted_model_file_name)
    
    
    args = dict(model="best.pt", data="A_my_data.yaml", epochs=100,save = False)
    trainer = DetectionTrainer(overrides=args)
    trainer.setup_model()
    trainer.set_model_attributes()
    trainer = trainer.model.to(device)
    trainer.eval()

    
    train_dataloader, train_data_count = utils.load_mnist(
         batch_size=training_parameters["BATCH_SIZE"], shuffle=False)
   
    
    
    # Train the AdvGAN model
    
    
    advGAN = AdvGAN(device,V8,trainer, model_num_labels, image_nc,
                           BOX_MIN, BOX_MAX, training_parameters["LEARNING_RATE"],model_path=model_path,grad_path=grad_path)
    
    
    history = advGAN.train(train_dataloader, training_parameters["EPOCHS"])
   

    # Plots
    utils.plot_performance(history["counter"],
                           data=[history["disc_losses"], history["gen_losses"],
                                 history["perturb_losses"], history["loss"]],
                           plt_names=["discriminator's mse loss", "generator's mse loss",
                                      "perturbation's loss", "loss"],
                           fig_name="GAN_model_performance",
                           y_name="loss")
    utils.plot_performance(history["counter"],
                           data=[history["disc_losses"], history["gen_losses"]],
                           plt_names=["discriminator's mse loss", "generator's mse loss"],
                           fig_name="discriminator_generator_GAN_model_performance",
                           y_name="mse loss")
    utils.plot_performance(history["counter"],
                           data=[history["perturb_losses"]],
                           plt_names=["perturbation's loss"],
                           fig_name="perturbation_GAN_model_performance",
                           y_name="perturbation's loss",
                           colors=['mediumvioletred'])
    utils.plot_performance(history["counter"],
                           data=[history["loss"]],
                           plt_names=["loss"],
                           fig_name="loss_GAN_model_performance",
                           y_name="adversarial's loss",
                           colors=['crimson'])
