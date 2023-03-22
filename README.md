# DFFUZZ

## Description

`DFFuzz` is a novel regression fuzzing framework for deep learning systems. It is designed to generate high-fidelity test inputs that trigger diverse regression faults effectively. To improve the fault-triggering capability of test inputs, `DFFuzz` adopts a Markov Chain Monte Carlo (MCMC) strategy to select mutation rules that are prone to trigger regression faults. Furthermore, to enhance the diversity of generated test inputs, we propose a diversity criterion to guide triggering more faulty behaviors. In addition, `DFFuzz` incorporates a GAN-based fidelity assurance method to guarantee the fidelity of test inputs. We conducted an empirical study to evaluate the effectiveness of `DFFuzz` on four regression scenarios (i.e., supplementary training, adversarial training, model fixing, and model pruning). The experimental results demonstrate the effectiveness of `DFFuzz`.



## The Structure

Here, we briefly introduce the usage/function of each directory: 

- `gan`: the GAN-based Fidelity Assurance Technique (the GAN structure and prediction code)
- `dcgan`: the DCGAN-based Fidelity Assurance Technique (the DCGAN structure and prediction code)
- `acgan`: the ACGAN-based Fidelity Assurance Technique (the ACGAN structure and prediction code)
- `models`: the original models and its regression model. 
- `params`: some params of `DFFuzz` and each model/datasets
- `utils`: the tools of `DFFuzz` & The experimental script and the input evaluation to calculate the diversity in `DFFuzz`


## Datasets/Models

We use 4 popular DL models based on 4 datasets under 2 regression scenarios, as the initial seed models in `DFFuzz`, which have been widely used in many existing studies.

| ID   | model    | dataset | M1_acc | M2_acc | Scenario |
| ---- | -------- | ------- | ------ | ------ | -------- |
| 1    | LeNet5   | MNIST   | 85.87% | 97.83% | SUPPLY   |
| 2    | LeNet5   | MNIST   | 98.07% | 98.30% | ADV      |
| 3    | VGG16    | CIFAR10 | 87.67% | 87.88% | SUPPLY   |
| 4    | VGG16    | CIFAR10 | 87.92% | 88.00% | ADV      |
| 5    | AlexNet  | FM      | 89.33% | 90.34% | SUPPLY   |
| 6    | AlexNet  | FM      | 91.70% | 91.87% | ADV      |
| 7    | ResNet18 | SVHN    | 88.85% | 91.93% | SUPPLY   |
| 8    | ResNet18 | SVHN    | 92.05% | 92.01% | ADV      |



1: We design 2 regression scenarios: supplementary training (denoted as SUPPLY), adversarial training (denoted as ADV).

2: For SUPPY, we select 80% of the training set to train the original model and use the 20% remaining data for supplementary training.

3: For ADV, we provide adversarial training on C&W adversarial examples.

4: If you want to download other models, please check the link below:
- MNIST: https://drive.google.com/file/d/1l_s--uSm5TN5a0xUTb1WobRHLti6o7Zw/view?usp=sharing
- Cifar10:https://drive.google.com/file/d/14ZBdd_AlDfVYcbdV31O0MNFHh0-z87cL/view?usp=sharing
- FM:https://drive.google.com/file/d/1C-gl_HgOOirM4I1mhDvShKFGLPzFlRbn/view?usp=sharing
- SVHN:https://drive.google.com/file/d/1UFZv6WZ0b-W0Qk4o8xfvAhK-mPiJhQGk/view?usp=sharing
After downloading the models, please put these models in `models` and make sure that the address of models are correct in **src/experiment_builder.py**. 

## The Requirements:

- python==3.7  (In fact 3.7.5 and 3.7.16 fits our work)

- keras==2.3.1 

- tensorflow==1.15.0 

- cleverhans==3.0.1  

- h5py==2.10.0

Please note that if you encounter the following error 'TypeError: Descriptors can not be created directly.', you may need to downgrade the protobuf package to 3.20.x or lower. It is so rarely happened. You can use 'pip install protobuf==3.20.1' to avoid this circumstance. Still, if you are confused with the environment setting, we also provided a file named `requirement.txt` to facilitate the installation process. You can directly use the script below. You can choose to use _pip_ or _conda_ in the script. 

~~~
pip install -r requirement.txt
~~~

We strongly suggest you run this project on **Linux**. We implemented the entire project on **Linux version 4.18.0-15-generic**. We will provide the configuration on windows in this project description in the future.


## Reproducibility

### Environment

We conducted 8 experiments in `DFFuzz`. The basic running steps are presented as follows:

**Step 0:** Please install the above runtime environment.

**Step 1:** Clone this repository. Download the dataset and models from our Google-Drive. 
Save the code and unzip datasets and models to `/your/local/path/`, e.g., `/your/local/path/models` and `/your/local/path/dataset`. 
(`/your/local/path/` should be the absolute path on your server, e.g., `/home/user_xxx/`) 

**Step 2:** Train yourself DCGAN models and save them to `/your/local/path/dcgan`. (Or you can use the one provided by us for reproductivity.)

**Step 3:** Edit configuration files `/your/local/path/experiment_builder.py` and `/your/local/path/dcgan/DCGAN_utils.py` in order to set the dataset, model, and DCGAN model path into `DFFuzz`

### Running DFFuzz

The `DFFuzz` artifacts are well organized, and researchers can simply run `DFFuzz` with the following command.

~~~
python run_dffuzz.py --params_set mnist LeNet5 change dffuzz --dataset MNIST --model LeNet5 --coverage change --time 1440
~~~

run_dffuzz.py contains all the configurations for our experimentation.

`--params_set` refers to the configuration and parameters in each dataset/model. Please select according to your requirements based on files in `params` directory. 
            If you want to adopt your own parameters, please go to `your/local/path/params` to change the setting params.

`--dataset` refers to the dataset you acquired. There are totally 4 choices ["MNIST", "CIFAR10", "FM", "SVHN"]

`--models` refers to the model you acquired. We totally designed 8 models according to 'Datasets/Models'. The settings are presented in the run_dffuzz.py. 
            Please select according to Datasets/Models and your experimental settings.

`--coverage` refers to the coverage used guide the fuzzing process; please set it to 'change' if you want to use `DFFuzz`; other choices are for compared approaches such as DeepHunter.

`--time` refers to the time of running our experiments. We set it to 1440 minutes (24 hours) for our experiment. For quick installation, you can set it to 5 minutes as a try.

## Reusability

For users who want to evaluate if `DFFuzz` is effective on their own regression models, we also prepared an approach to reuse `DFFuzz` on new regression models and new datasets.

Firstly, the users need to update the addresses of their own datasets and regression models under test in function _\_get\_dataset_ and function _\_get\_models_ in the **experiment\_builder.py** file for `DFFuzz` to load. Please note that if the dataset requires further preprocessing, the users should also update the corresponding preprocessing method _picture\_preprocess_ in the **src/utility.py** file. 

Secondly, the users are required to train a simple Discriminator of GAN (e.g., in `dcgan`) to guarantee the fidelity of generated test inputs. From that, `DFFuzz` can be adapted to new regression models under test; it will conduct the following fuzzing process and finish the entire job. 

Please do not forget to name your own regression scenarios and regression datasets, setting the corresponding parameters (or configurations) in `params` so that you can load the parameters through experimental scripts.
