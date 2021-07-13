<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Medium][medium-shield]][medium-url]
[![Twitter][twitter-shield]][twitter-url]
[![Linkedin][linkedin-shield]][linkedin-url]

# SHAP: Shapley Additive Explanations
This repository contains an example on how to implement the `shap` library to interpret a machine learning model.

If you want to know in detail how SHAP works, what its components are and how to interpret ML models with the `shap` library, I recommend you to take a look at the article: <a href="https://towardsdatascience.com/shap-shapley-additive-explanations-5a2a271ed9c3"> SHAP: Shapley Additive Explanations </a>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [How to use](#how-to-use)
* [Contact](#contact)
* [License](#license)

<!-- how-to-use -->
## 1. How to use
There are 2 steps that you should follow, first you have to preprocess the dataset by typing:

```py
from src.preprocess import Data
data = Data(csv_path='data/prostate_cancer.csv')
```

then, for training and optimizing the classifier, you need to type:


```PY
from src.model import Classifier
classifier = Classifier(x_train=data.x_train, x_test=data.x_test, y_train=data.y_train, y_test=data.y_test)
```

It is important to mention that the approach shown in this repository is aligned with the examples shown in the SHAP article: **Shapley Additive Explanation**. Also, for ease you can use the demo implemented in the jupyter notebook ``shap_demo.ipynb``.

<!-- contributing -->
## 4. Contributing
Feel free to fork the model and add your own suggestiongs.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourGreatFeature`)
3. Commit your Changes (`git commit -m 'Add some YourGreatFeature'`)
4. Push to the Branch (`git push origin feature/YourGreatFeature`)
5. Open a Pull Request

<!-- contact -->
## 5. Contact
If you have any question, feel free to reach me out at:
* <a href="https://twitter.com/Fernando_LpzV">Twitter</a>
* <a href="https://ferneutron.medium.com/>Medium</a>
* <a href="https://www.linkedin.com/in/fernando-lopezvelasco/">Linkedin</a>
* Email: fer.neutron@gmail.com

<!-- license -->
## 6. License
Distributed under the MIT License. See ``LICENSE.md`` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[medium-shield]: https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white
[medium-url]: https://ferneutron.medium.com/
[twitter-shield]: https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white
[twitter-url]: https://twitter.com/Fernando_LpzV
[linkedin-shield]: https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://www.linkedin.com/in/fernando-lopezvelasco/