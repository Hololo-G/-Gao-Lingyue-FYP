# -Gao-Lingyue-FYP
This GitHub project is created for the Final Year Project of Gao Lingyue SBC-20-1002 935547 from University of Shanghai for Science and Technology Sino-British College and Liverpool John Moores University. They are the codes about classification part, including data preprocessing, VGG16 training, VGG16 with SE block training, model testing, and GUI. data_preprocessing.py contains all the augmentation steps. dataset_enhanced is the data after preprocessing, which can be trained directly. In VGG16.py, researchers can change the value of batch size and initial learning rate to compare the performance. In VGG16_SE_Block.py, a SE module is added after the last fully connected layer of pre-trained VGG-16. Testing.py output some evaluation metrics. GUI.py builds a GUI window to facilitate application.
