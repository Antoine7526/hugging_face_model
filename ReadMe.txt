PROJET FOUILLE DE TEXTES : Fouille d'opinions dans les commentaires de clients

MODÈLE
Contexte :
Dans le cadre de ce projet, nous avons fait le choix d'utiliser un Transformer et plus particulièrement le modèle
pré-entraîné Yanzhu/bertweetfr-base provenant du hub HuggingFace (https://huggingface.co/Yanzhu/bertweetfr-base). Ce
modèle utilise la même architecture que CamemBERT-base (la version basique deCamemBERT). La majeure différence est que
ce modèle a été entraîné sur 15Go de tweets français.

Architecture du modèle :
CamemBERT-base est composé :
    - d'une couche d'embedding, pour représenter chaque mot en vecteur
    - de douze couches cachées composées principalement de deux types de transformations, des transformations dites
      self-attention et des transformations denses

Dans la plupart des couches, la taille des vecteurs est de 768 (cf. model.hidden_size).

Hyperparamètres du modèle :
Nous avons utilisé l'optimiseur AdamW (disponible avec torch.optim), car il est très utilisé avec les transformers
et parce qu'il s'est avéré plus performant que l'optimiseur Adam.
Afin de l'optimiser au mieux, nous lui avons passé en paramètre le learning rate (ou taux d'apprentissage), qui permet
d'ajuster la longueur du pas à chaque étape d'optimisation, l'epsilon, permettant d'éviter les divisions par zéro et le
weight decay, une méthode de Regularization qui va diminuer les poids, ce qui permet d'améliorer la généralisation
lors de l'apprentissage.

Pour continuer, nous avons ajouté un dropout dans la configuration du modèle. Cette méthode de Regularization a pour
but "d'éteindre" temporairement un pourcentage de neurones dans les couches d'un modèle. Cela permet de réduire
l'overfitting.

Après de nombreuses expérimentations, nous avons retenu pour nos hyperparamètres les valeurs suivantes :
    - le learning rate (HP.lr) : 1e-5
    - le weight decay (HP.weight_decay) : 1e-6
    - l'epsilon pour l'optimiseur AdamW (HP.ad_eps) : 1e-5
    - le dropout : 0.2

Performances du modèle :
Sur les données de développements, nous trouvons une moyenne de 83.88% de taux de bonne classification
(sur les 25 derniers runs) avec un temps moyen de 20 minutes (pour 5 runs avec les gpu Google Colab).

DÉPENDANCES
Pour faire tourner classifier.py dans Google Colab, vous aurez besoin des lignes ci-dessous :
    !pip install pytorch_lightning
    !pip install transformers
    !pip install sklearn
    !pip install sentencepiece (car CamemBERT-base l'embarque)

AUTEURS
Batiste Amistadi & Antoine Grancher

CONTACTS
antoine.grancher@etu.univ-grenoble-alpes.fr
batiste.amistadi@etu.univ-grenoble-alpes.fr

