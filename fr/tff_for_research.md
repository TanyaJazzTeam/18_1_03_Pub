# Utiliser TFF pour la recherche sur l'apprentissage fédéré

## Aperçu ds

TFF est un cadre extensible et puissant permettant de mener des recherches sur l'apprentissage fédéré (FL) en simulant des calculs fédérés sur des ensembles de données proxy réalistes. Cette page décrit les principaux concepts et composants pertinents pour les simulations de recherche, ainsi que des conseils détaillés pour mener différents types de recherche dans TFF.

## La structure typique du code de recherche dans TFFdearf

Une simulation FL de recherche implémentée dans TFF se compose généralement de trois types principaux de logique.

1. Des morceaux individuels de code TensorFlow, généralement `tf.function` s, qui encapsulent une logique qui s'exécute à un seul emplacement (par exemple, sur des clients ou sur un serveur). Ce code est généralement écrit et testé sans aucune référence `tff.*` et peut être réutilisé en dehors de TFF. Par exemple, la [boucle de formation client dans Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) est implémentée à ce niveau.
2. Logique d'orchestration fédérée TensorFlow, qui lie les `tf.function` s individuels de 1. en les enveloppant sous forme de `tff.tf_computation` s, puis en les orchestrant à l'aide d'abstractions telles que `tff.federated_broadcast` et `tff.federated_mean` dans un `tff.federated_computation` . Voir, par exemple, cette [orchestration pour Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140) .
3. Un script de pilote externe qui simule la logique de contrôle d'un système FL de production, en sélectionnant des clients simulés à partir d'un ensemble de données, puis en exécutant les calculs fédérés définis en 2. sur ces clients. Par exemple, [un pilote d'expérience EMNIST fédéré](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) .

## Ensembles de données d'apprentissage fédéré ewt

TensorFlow fédéré [héberge plusieurs ensembles de données](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets) représentatifs des caractéristiques de problèmes du monde réel qui pourraient être résolus grâce à l'apprentissage fédéré.

Remarque : Ces ensembles de données peuvent également être consommés par n'importe quel framework ML basé sur Python sous forme de tableaux Numpy, comme documenté dans l' [API ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData) . ewt

Les ensembles de données comprennent :

- [**StackOverflow** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) Un ensemble de données textuelles réalistes pour la modélisation du langage ou les tâches d'apprentissage supervisé, avec 342 477 utilisateurs uniques et 135 818 730 exemples (phrases) dans l'ensemble de formation.
- [**EMNISTE Fédéré** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) Un prétraitement fédéré de l'ensemble de données de caractères et de chiffres EMNIST, où chaque client correspond à un écrivain différent. La rame complète contient 3 400 utilisateurs avec 671 585 exemples provenant de 62 étiquettes.
- [**Shakespeare** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data) Un ensemble de données textuelles plus petit au niveau des caractères, basé sur les œuvres complètes de William Shakespeare. L'ensemble de données se compose de 715 utilisateurs (personnages de pièces de Shakespeare), où chaque exemple correspond à un ensemble contigu de lignes prononcées par le personnage dans une pièce donnée.
- [**CIFAR-100** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) Un partitionnement fédéré de l'ensemble de données CIFAR-100 sur 500 clients de formation et 100 clients de test. Chaque client dispose de 100 exemples uniques. Le cloisonnement est réalisé de manière à créer une hétérogénéité plus réaliste entre les clients. Pour plus de détails, consultez l' [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) .
- [**Ensemble de données Google Landmark v2** L'ensemble de données se compose de photos de divers monuments du monde, avec des images regroupées par photographe pour obtenir un partitionnement fédéré des données. Deux types d'ensembles de données sont disponibles : un ensemble de données plus petit avec 233 clients et 23 080 images, et un ensemble de données plus grand avec 1 262 clients et 164 172 images.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data)
- [**CelebA** Un ensemble de données d'exemples (image et attributs faciaux) de visages de célébrités. L'ensemble de données fédéré regroupe les exemples de chaque célébrité pour former un client. Il y a 9343 clients, chacun avec au moins 5 exemples. L'ensemble de données peut être divisé en groupes d'entraînement et de test soit par clients, soit par exemples.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data)
- [**iNaturalist** Un ensemble de données se compose de photos de diverses espèces. L'ensemble de données contient 120 300 images pour 1 203 espèces. Sept versions de l'ensemble de données sont disponibles. L'un d'eux est regroupé par photographe et comprend 9257 clients. Le reste des ensembles de données est regroupé par emplacement géographique où la photo a été prise. Ces six versions de l'ensemble de données comprennent 11 à 3 606 clients.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data)

## Simulations hautes performances etw

Bien que le temps d'horloge d'une *simulation* FL ne soit pas une mesure pertinente pour évaluer les algorithmes (car le matériel de simulation n'est pas représentatif des environnements de déploiement FL réels), être capable d'exécuter rapidement des simulations FL est essentiel pour la productivité de la recherche. Par conséquent, TFF a investi massivement dans la fourniture d’environnements d’exécution hautes performances sur une ou plusieurs machines. La documentation est en cours de développement, mais pour l'instant, consultez les instructions sur [les simulations TFF avec des accélérateurs](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators) et les instructions sur [la configuration des simulations avec TFF sur GCP](https://www.tensorflow.org/federated/gcp_setup) . Le runtime TFF hautes performances est activé par défaut.

## TFF pour différents domaines de recherche

### Algorithmes d'optimisation fédérés

La recherche sur les algorithmes d'optimisation fédérés peut être effectuée de différentes manières dans TFF, selon le niveau de personnalisation souhaité.

Une implémentation autonome minimale de l'algorithme [Federated Averaging](https://arxiv.org/abs/1602.05629) est fournie [ici](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg) . Le code comprend [des fonctions TF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py) pour le calcul local, [des calculs TFF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) pour l'orchestration et un [script de pilote](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) sur l'ensemble de données EMNIST à titre d'exemple. Ces fichiers peuvent facilement être adaptés pour des applications personnalisées et des changements algorithmiques en suivant les instructions détaillées du [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md) .

Une implémentation plus générale de Federated Averaging peut être trouvée [ici](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py) . Cette implémentation permet des techniques d'optimisation plus sophistiquées, notamment l'utilisation de différents optimiseurs à la fois sur le serveur et sur le client. D'autres algorithmes d'apprentissage fédéré, y compris le clustering fédéré à k-moyennes, peuvent être trouvés [ici](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/) .drwrwrf wrfrf

### Compression de mise à jour du modèle

Une compression avec perte des mises à jour du modèle peut entraîner une réduction des coûts de communication, ce qui peut entraîner une réduction du temps de formation global.

Pour reproduire un [article](https://arxiv.org/abs/2201.02664) récent, voir [ce projet de recherche](https://github.com/google-research/federated/tree/master/compressed_communication) . Pour implémenter un algorithme de compression personnalisé, consultez [comparative_methods](https://github.com/google-research/federated/tree/master/compressed_communication/aggregators/comparison_methods) dans le projet pour les lignes de base à titre d'exemple, et [le didacticiel TFF Aggregators](https://www.tensorflow.org/federated/tutorials/custom_aggregators) si vous n'êtes pas déjà familier.

### Confidentialité différentielle

TFF est interopérable avec la bibliothèque [TensorFlow Privacy](https://github.com/tensorflow/privacy) pour permettre la recherche de nouveaux algorithmes pour la formation fédérée de modèles avec confidentialité différentielle. Pour un exemple de formation avec DP à l'aide [de l'algorithme de base DP-FedAvg](https://arxiv.org/abs/1710.06963) et [de ses extensions](https://arxiv.org/abs/1812.06210) , consultez [ce pilote d'expérience](https://github.com/google-research/federated/blob/master/differential_privacy/stackoverflow/run_federated.py) .

Si vous souhaitez implémenter un algorithme DP personnalisé et l'appliquer aux mises à jour globales de la moyenne fédérée, vous pouvez implémenter un nouvel algorithme de moyenne DP en tant que sous-classe de [](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) `tensorflow_privacy.DPQuery` et créez un `tff.aggregators.DifferentiallyPrivateFactory` avec une instance de votre requête. Un exemple d'implémentation de l' [algorithme DP-FTRL](https://arxiv.org/abs/2103.00039) peut être trouvé [](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

Les GAN fédérés (décrits [ci-dessous](#generative_adversarial_networks) ) sont un autre exemple de projet TFF mettant en œuvre une confidentialité différentielle au niveau de l'utilisateur (par exemple, [ici dans le code](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L144) ).

### Robustesse et attaques

TFF peut également être utilisé pour simuler les attaques ciblées sur les systèmes d'apprentissage fédéré et les défenses différentielles basées sur la confidentialité envisagées dans *[Can You Really Back door Federated Learning ?](https://arxiv.org/abs/1911.07963) . Cela se fait en construisant un processus itératif avec des clients potentiellement malveillants (voir [](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L412) `build_federated_averaging_process_attacked` ). Le [répertoire contient plus de détails.](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack)*

- De nouveaux algorithmes d'attaque peuvent être implémentés en écrivant une fonction de mise à jour client qui est une fonction Tensorflow, voir [`ClientProjectBoost` pour un exemple.](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460)
- De nouvelles défenses peuvent être implémentées en personnalisant [« tff.utils.StatefulAggregateFn »](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103) qui regroupe les sorties client pour obtenir une mise à jour globale.

Pour un exemple de script de simulation, voir [`emnist_with_targeted_attack.py` .](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/emnist_with_targeted_attack.py)

### Réseaux adverses génératifs

Les GAN constituent un [modèle d'orchestration fédérée](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L266-L316) intéressant qui semble un peu différent de la moyenne fédérée standard. Ils impliquent deux réseaux distincts (le générateur et le discriminateur) chacun entraîné avec sa propre étape d'optimisation.

TFF peut être utilisé pour la recherche sur la formation fédérée des GAN. Par exemple, l'algorithme DP-FedAvg-GAN présenté dans [des travaux récents](https://arxiv.org/abs/1911.06679) est [implémenté dans TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans) . Ce travail démontre l'efficacité de la combinaison de l'apprentissage fédéré, des modèles génératifs et [de la confidentialité différentielle](#differential_privacy) .

### Personnalisation

La personnalisation dans le cadre de l’apprentissage fédéré est un domaine de recherche actif. L'objectif de la personnalisation est de fournir différents modèles d'inférence à différents utilisateurs. Il existe potentiellement différentes approches pour résoudre ce problème.

Une approche consiste à laisser chaque client affiner un modèle global unique (entraîné à l'aide de l'apprentissage fédéré) avec ses données locales. Cette approche a des liens avec le méta-apprentissage, voir, par exemple, [cet article](https://arxiv.org/abs/1909.12488) . Un exemple de cette approche est donné dans [`emnist_p13n_main.py` . Pour explorer et comparer différentes stratégies de personnalisation, vous pouvez :](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py)

- Définissez une stratégie de personnalisation en implémentant une `tf.function` qui part d'un modèle initial, entraîne et évalue un modèle personnalisé à l'aide des ensembles de données locaux de chaque client. Un exemple est donné par [`build_personalize_fn` .](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py)
- Définissez un `OrderedDict` qui mappe les noms de stratégie aux stratégies de personnalisation correspondantes et utilisez-le comme argument `personalize_fn_dict` dans [`tff.learning.build_personalization_eval` .](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)

Une autre approche consiste à éviter de former un modèle entièrement global en formant une partie d'un modèle entièrement localement. Une instanciation de cette approche est décrite dans [cet article de blog](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html) . Cette approche est également liée au méta-apprentissage, voir [cet article](https://arxiv.org/abs/2102.03448) . Pour explorer l’apprentissage fédéré partiellement local, vous pouvez :

- Consultez le [didacticiel](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) pour un exemple de code complet appliquant la reconstruction fédérée et [des exercices de suivi](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations) .
- Créez un processus de formation partiellement local à l'aide de [`tff.learning.reconstruction.build_training_process` , en modifiant `dataset_split_fn` pour personnaliser le comportement du processus.](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process)
