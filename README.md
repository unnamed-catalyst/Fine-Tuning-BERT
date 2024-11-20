# Fine-Tuning Transformer Models for Sentiment Analysis on Twitter Data

Code for a comparative analysis of the performance of fine-tuned transformer models on climate change data. The transformer models used were BERT, DistilBERT and RoBERTa.

## System Specifications 

The experiments were conducted on the following system:

- Processor: Intel Core i5-10500H CPU @ 2.50GHz
- Graphics Card: NVIDIA GTX 1650 Max-Q
- RAM: 16GB DDR4
- Storage: 512GB SSD
- Operating System: Windows 10
- Python Version: 3.11.5
- Frameworks: PyTorch 2.2.0, Transformers 4.33.2

## Dataset

The dataset used is a Twitter dataset focused on climate change, containing 43,943 tweets annotated into "Pro", "Anti", "Neutral", and "News". The dataset can be found on Kaggle at this [link](https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset).

The dataset was initially cleaned using the NeatText Python package before using it to fine-tune the transformer models, the process of data cleaning can be found in [Cleaning Data.ipynb](https://github.com/unnamed-catalyst/Fine-Tuning-BERT/blob/main/Cleaning%20Data.ipynb). The NeatText package can be found at this [link](https://github.com/Jcharis/neattext/wiki)

## Hyperparameters Used

<br>
<div align="center">
  <table>
    <tr>
      <th> Parameter </th> 
      <th> Description </th>
      <th> Value </th>
    </tr>
    <tr>
      <th> Epochs </th> 
      <td> As transformer models converge quickly, a low number of epochs was used </td>
      <td> 2 </td>
    </tr>
    <tr>
      <th> Batch Size </th> 
      <td> Due to memory limitations, a smaller batch size was used </td>
      <td> 8 </td>
    </tr>
    <tr>
      <th> Gradient Accumulation </th> 
      <td> Gradient accumulation over 2 steps was used to simulate a larger batch size </td>
      <td> 2 </td>
    </tr>
    <tr>
      <th> Learning Rate </th> 
      <td> As transformer models are sensitive to the learning rate, a small value was used </td>
      <td> 2e-5 </td>
    </tr>
    <tr>
      <th> Weight Decay </th> 
      <td> Regularization term used to prevent overfitting </td>
      <td> 0.01 </td>
    </tr>
    <tr>
      <th> Evaluation Strategy </th> 
      <td> Performance was evaluated after a set number of steps instead of after each epoch </td>
      <td> steps </td>
    </tr>
    <tr>
      <th> Evaluation Steps </th> 
      <td> The number of steps before the performance was evaluated with the validation set </td>
      <td> 805 </td>
    </tr>
    <tr>
      <th> Warmup Steps </th> 
      <td> Initial training steps where the learning rate gradually increases to the defined value </td>
      <td> 805 </td>
    </tr>
  </table>
Table 1: Hyperparameters Used for the Transformer Models
</div>
<br>

## Results

Key findings from the project:

- The **BERT** model outperformed the other transformer models on the given dataset with an accuracy of **90%**.
- The **DistilBERT** and **RoBERTa** models achieved accuracies of **88%** and **87%** respectively.
- The **Ensemble Model** outperformed all the models individually with an accuracy of **93.37%** on the given dataset.

<br>
<div align="center">
  <table>
    <tr>
      <th> Category </th> 
      <th> BERT </th>
      <th> DistilBERT </th>
      <th> RoBERTa </th>
      <th> Ensemble </th>
    </tr>
    <tr>
      <th> Pro </th> 
      <td> 0.88 </td>
      <td> 0.85 </td>
      <td> 0.82 </td>
      <td> 0.97 </td>
    </tr>
    <tr>
      <th> News </th> 
      <td> 0.94 </td>
      <td> 0.93 </td>
      <td> 0.93 </td>
      <td> 0.95 </td>
    </tr>
    <tr>
      <th> Anti </th> 
      <td> 0.97 </td>
      <td> 0.95 </td>
      <td> 0.95 </td>
      <td> 0.96 </td>
    </tr>
    <tr>
      <th> Neutral </th> 
      <td> 0.81 </td>
      <td> 0.78 </td>
      <td> 0.77 </td>
      <td> 0.85 </td>
    </tr>
    <tr>
      <th> Overall </th> 
      <td> 0.90 </td>
      <td> 0.88 </td>
      <td> 0.87 </td>
      <td> 0.93 </td>
    </tr>
  </table>
Table 2: Accuracies of the Various Transformer Models and the Ensemble Approach
</div>
<br>
<div align="center"> <img src="https://github.com/user-attachments/assets/b75f72af-6d47-482c-8b66-cdad63c207ea" alt="Transformer Model Architecture"> <p><b>Figure 1:</b> Ensemble Model Confusion Matrix</p> </div>

