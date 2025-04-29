# csci5525---homework-3-solved
**TO GET THIS SOLUTION VISIT:** [CSCI5525 – Homework 3 Solved](https://www.ankitcodinghub.com/product/csci5525-homework-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;117048&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSCI5525 - Homework 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
1. (10 points) Consider the convolutional neural network architecture given in Figure 1 for classifying MNIST digits. Assume that the input images are reduced to size 10 × 10 with only 1 channel (represented as a matrix in R10×10). In this architecture, the convolutional layer uses a 3 × 3 filter, Wconv, with stride 2 and zero padding stride 3 and zero padding of size 1. The dimensions of the outputs of each layer are shown below.

Figure 1: A toy CNN

(a) (5 points) What are the values of n,m,k in the graph?

(b) (5 points) What are the sizes of Wconv, Wfc, and b?

Programming Problems: The next two problems are programming problems and will focus on implementing neural networks for handwritten digits classification. We will use the MNIST dataset where each sample is an image of a hand-written digit and has a corresponding label indicating the value of the digit written (0,1,…,9). This makes it a multi-class classification problem.

You must use Tensorflow 2 to implement your neural networks. The implementation must be for the CPU version only (no GPUs or MPI parallel programming is required for this assignment).

import tensorflow as tf mnist = tf.keras.datasets.mnist

(x train, y train), (x test, y test) = mnist.load data()

2. (40 points) Implement a multi-layer fully connected neural network:

• Input: 1-channel input, size 28×28

• Fully connected layer 1: input with bias; output – 128 nodes

• ReLU activation function

• Fully connected layer 2: input – 128 nodes; output – 10 nodes

• Softmax activation function

• Use cross entropy as the loss function

• Use SGD as optimizer

• Set mini batch size as 32

Train using mini batches of the given batch size. Plot the cumulative training loss and accuracy for every epoch. Once training is complete, apply the learned model to the test set and report the testing accuracy.

Epoch: An epoch is a single pass through all the training data. Typically many epochs will be run when training a neural network before it converges.

Please submit (a) summary of methods and results report and (b) code:

(a) Summary of methods and results: Briefly describe the approaches used above, along with relevant equations. Report the cumulative training loss and accuracy for every epoch as plots. Also report the testing accuracy (a single number).

(b) Code: Submit the file neural net.py which contains the function def neural net() -&gt; None:. The function does not have any inputs and does not return anything but must print out to the terminal (stdout) the cumulative training loss and accuracy per epoch as well as the testing accuracy.

3. (50 points) Implement a convolutional neural network with the following specifications.

• Input: 1-channel input, size 28×28

• Convolution layer: Convolution kernel size is (3, 3) with stride as 1. Input channels – 1; Output channels – 20 nodes

• ReLU activation function

• Max-pool: 2×2 max pool

• Dropout layer with probability p = 0.50

• Flatten input for feed to fully connected layers

• Fully connected layer 1: flattened input with bias; output – 128 nodes

• ReLU activation function

• Dropout layer with probability p = 0.50

• Fully connected layer 2: input – 128 nodes; output – 10 nodes

• Softmax activation function

• Use cross entropy as loss function

For this problem, we will be experimenting with a variety of parameters.

First, train using SGD as the optimizer and mini batches of size 32. Plot the cumulative training loss and accuracy for every epoch. Once training is complete, apply the learned model to the test set and report the testing accuracy.

Second, train your network using mini batch sizes of [32,64,96,128] and plot the convergence run time vs mini batch sizes for each of the following optimizers: SGD, Adagrad, and Adam. You should report 3 figures, one for each optimizer where each figure has mini batch size on the x-axis and the convergence run time on the y-axis.

Please submit (a) summary of methods and results report and (b) code:

(a) Summary of methods and results: Briefly describe the approaches used above, along with relevant equations. Report the cumulative training loss and accuracy for every epoch as plots. Also report the testing accuracy (a single number). Also report the convergence run time vs mini batch sizes for each mini batch size and optimizer above (3 plots).

(b) Code: Submit the file cnn.py which contains the function def cnn() -&gt; None:. The function does not have any inputs and does not return anything but must print out to the terminal (stdout) the cumulative training loss and accuracy as well as the testing accuracy (for the first part above). It must also print out the batch size and convergence run time for each mini batch size and optimizer (for the second part above).

Additional instructions: Code can only be written in Python 3.6+; no other programming languages will be accepted. One should be able to execute all programs from the Python command prompt or terminal. Please specify instructions on how to run your program in the README file.

Each function must take the inputs in the order specified in the problem and display the textual output via the terminal and plots/figures should be included in the report.

Your code must be runnable on a CSE lab machine (e.g., csel-kh1260-01.cselabs.umn.edu). One option is to SSH into a machine. Learn about SSH at these links: https://cseit.umn.edu/ knowledge-help/learn-about-ssh, https://cseit.umn.edu/knowledge-help/choose-ssh-tool, and https://cseit.umn.edu/knowledge-help/remote-linux-applications-over-ssh.

Instructions

Follow the rules strictly. If we cannot run your code, you will not get any credit.

• Things to submit

1. hw3.pdf: The report that contains the solutions to Problems 1, 2, and 3 including the summary of methods and results.

2. neural net.py: Code for Problem 2.

3. cnn.py: Code for Problem 3.

4. README.txt: README file that contains your name, student ID, email, instructions on how to run your code, any assumptions you are making, and any other necessary details.

5. Any other files, except the data, which are necessary for your code.

Homework Policy. (1) You are encouraged to collaborate with your classmates on homework problems, but each person must write up the final solutions individually. You need to list in the README.txt which problems were a collaborative effort and with whom. (2) Regarding online resources, you should not:

• Google around for solutions to homework problems,

• Ask for help on online,

• Look up things/post on sites like Quora, StackExchange, etc.
