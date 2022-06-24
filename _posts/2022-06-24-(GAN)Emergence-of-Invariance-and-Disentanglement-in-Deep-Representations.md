---
layout: post 
title: "(GAN)Emergence of Invariance and Disentanglement in Deep Representation Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

### [$$\mathbf{Emergence\;of\;Invariance\;and\;Disentanglement\;}$$](https://arxiv.org/pdf/1706.01350.pdf)

### [$$\mathbf{in\;Deep\;Representations}$$](https://arxiv.org/pdf/1706.01350.pdf)

#### $$\mathbf{Alessandro\;chille}$$

#### $$\mathbf{Stefano\;Soatto}$$

### $\mathbf{Abstract}$

> Using established principles from Statistics and Information Theory, we show that invariance to nuisance factors in a deep neural network is equivalent to information minimality of the learned representation, and that stacking layers and injecting noise during training naturally bias the network towards learning invariant representations. We then decompose the cross-entropy loss used during training and highlight the presence of an inherent overfitting term. We propose regularizing the loss by bounding such a term in two equivalent ways: One with a Kullbach-Leibler term, which relates to a PAC-Bayes perspective; the other using the information in the weights as a measure of complexity of a learned model, yielding a novel Information Bottleneck for the weights. Finally, we show that invariance and independence of the components of the representation learned by the network are bounded
above and below by the information in the weights, and therefore are implicitly optimized during training. The theory enables us to quantify and predict sharp phase transitions between underfitting and overfitting of random labels when using our regularized loss, which we verify in experiments, and sheds light on the relation between the geometry of the loss function, invariance properties of the learned representation, and generalization error.
Keywords: Representation learning; PAC-Bayes; information bottleneck; flat minima; generalization; invariance; independence;

### $\mathbf{1.\;Introduction}$

> Efforts to understand the empirical success of deep learning have followed two main lines: Representation learning and optimization. In optimization, a deep network is treated as a black-box family of functions for which we want to find parameters (weights) that yield good generalization. Aside from the difficulties due to the non-convexity of the loss function, the fact that deep networks are heavily over-parametrized presents a theoretical challenge: The bias-variance trade-off suggests they may severely overfit; yet, even without explicit regularization, they perform remarkably well in practice. Recent work suggests that this is related to properties of the loss landscape and to the implicit regularization performed by stochastic gradient descent (SGD), but the overall picture is still hazy (Zhang et al., 2017).

![Figure 1]()

> Figure 1: (Left) The AlexNet model of Zhang et al. (2017) achieves high accuracy (red) even when trained with random labels on CIFAR-10. Using the IB Lagrangian to limit information in the weights leads to a sharp transition to underfitting (blue) predicted by the theory (dashed line). To overfit, the network needs to memorize the dataset, and the information needed grows linearly. (Right) For real labels, the information sufficient to fit the data without overfitting saturates to a value that depends on the dataset, but somewhat independent of the number of samples. Test accuracy shows a uniform blue plot for random labels, while for real labels it increases with the number of training samples, and is higher near the critical regularizer value $\beta=1$.

> Representation learning, on the other hand, focuses on the properties of the representation learned by the layers of the network (the activations) while remaining largely agnostic to the particular optimization process used. In fact, the effectiveness of deep learning is often ascribed to the ability of deep networks to learn representations that are insensitive (invariant) to nuisances such as translations, rotations, occlusions, and also “disentangled,” that is, separating factors in the high-dimensional space of data (Bengio, 2009). Careful engineering of the architecture plays an important role in achieving insensitivity to simple geometric nuisance transformations, like translations and small deformations; however, more complex and dataset-specific nuisances still need to be learned. This poses a riddle: If neither the architecture nor the loss function explicitly enforce invariance and disentangling, how can these properties emerge consistently in deep networks trained by simple generic optimization?

> In this work, we address these questions by establishing information theoretic connections between these concepts. In particular, we show that: (a) a sufficient representation of the data is invariant if and only if it is minimal, i.e., it contains the smallest amount of information, although may not have small dimension; (b) the information in the representation, along with its total correlation (a measure of disentanglement) are tightly bounded by the information that the weights contain about the dataset; (c) the information in the weights, which is related to overfitting (Hinton and Van Camp, 1993), flat minima (Hochreiter and Schmidhuber, 1997), and a PAC-Bayes upper-bound on the test error (Section 6), can be controlled by implicit or explicit regularization. Moreover, we show that adding noise during the training is a simple and natural way of biasing the network towards invariant representations.

> Finally, we perform several experiments with realistic architectures and datasets to validate the assumptions underlying our claims. In particular, we show that using the information in the weights to measure the complexity of a deep neural network (DNN), rather than the number of its parameters, leads to a sharp and theoretically predicted transition between overfitting and underfitting regimes for random labels, shedding light on the questions of Zhang et al. (2017).

#### $\mathbf{1.1\;Related\;work}$

> The Information Bottleneck (IB) was introduced by Tishby et al. (1999) as a generalization of minimal sufficient statistics that allows trading off fidelity (sufficiency) and complexity of a representation. In particular, the IB Lagrangian reduces finding a minimal sufficient representation to a variational optimization problem. Later, Tishby and Zaslavsky (2015) and Shwartz-Ziv and Tishby (2017) advocated using the IB between the test data and the activations of a deep neural network, to study the sufficiency and minimality of the resulting representation. In parallel developments, the IB Lagrangian was used as a regularized loss function for learning representation, leading to new information theoretic regularizers (Achille and Soatto, 2018; Alemi et al., 2017a; Alemi et al., 2017b).

> In this paper, we introduce an IB Lagrangian between the weights of a network and the training data, as opposed to the traditional one between the activations and the test datum. We show that the former can be seen both as a generalization of Variational Inference, related to Hinton and Van Camp (1993), and as a special case of the more general PAC-Bayes framework (McAllester, 2013), that can be used to compute high-probability upper-bounds on the test error of the network. One of our main contributions is then to show that, due to a particular duality induced by the architecture of deep networks, minimality of the weights (a function of the training dataset) and of the learned representation (a function of the test input) are connected: in particular we show that networks regularized either explicitly, or implicitly by SGD, are biased toward learning invariant and disentangled representations. The theory we develop could be used to explain the phenomena described in small-scale experiments in Shwartz-Ziv and Tishby (2017), whereby the initial fast convergence of SGD is related to sufficiency of the representation, while the later asymptotic phase is related to compression of the activations: While SGD is seemingly agnostic to the property of the learned representation, we show that it does minimize the information in the weights, from which the compression of the activations follows as a corollary of our bounds. Practical implementation of this theory on real large scale problems is made possible by advances in Stochastic Gradient Variational Bayes (Kingma and Welling, 2014; Kingma et al., 2015).

> Representations learned by deep networks are observed to be insensitive to complex nuisance transformations of the data. To a certain extent, this can be attributed to the architecture. For instance, the use of convolutional layers and max-pooling can be shown to yield insensitivity to local group transformations (Bruna and Mallat, 2011; Anselmiet al., 2016; Soatto and Chiuso, 2016). But for more complex, dataset-specific, and in particular non-local, non-group transformations, such insensitivity must be acquired as part of the learning process, rather than being coded in the architecture. We show that a sufficient representation is maximally insensitive to nuisances if and only if it is minimal, allowing us to prove that a regularized network is naturally biased toward learning invariant representations of the data.

> Efforts to develop a theoretical framework for representation learning include Tishby and Zaslavsky (2015) and Shwartz-Ziv and Tishby (2017), who consider representations as stochastic functions that approximate minimal sufficient statistics, different from Bruna and Mallat (2011) who construct representations as (deterministic) operators that are invertible in the limit, while exhibiting reduced sensitivity (“stability”) to small perturbations of the data. Some of the deterministic constructions are based on the assumption that the underlying data is spatially stationary, and therefore work best on textures and other visual data that are not subject to occlusions and scaling nuisances. Anselmi et al. (2016) develop a theory of invariance to locally compact groups, and aim to construct maximal (“distinctive”) invariants, like Sundaramoorthi et al. (2009) that, however, assume nuisances to be infinite-dimensional groups (Grenander, 1993). These efforts are limited by the assumption that nuisances have a group structure. Such assumptions were relaxed by Soatto and Chiuso (2016) who advocate seeking for sufficient invariants, rather than maximal ones. We further advance this approach, but unlike prior work on sufficient dimensionality reduction, we do not seek to minimize the dimension of the representation, but rather its information content, as prescribed by our theory. Recent advances in Deep Learning provide us with computationally viable methods to train high-dimensional models and predict and quantify observed phenomena such as convergence to flat minima and transitions from overfitting to underfitting random labels, thus bringing the theory to fruition. Other theoretical efforts focus on complexity considerations, and explain the success of deep networks by ways of statistical or computational efficiency (Lee et al., 2017; Bengio, 2009; LeCun, 2012). “Disentanglement” is an often-cited property of deep networks (Bengio, 2009), but seldom formalized and studied analytically, although Ver Steeg and Galstyan (2015) has suggested studying it using the Total Correlation of the representation, also known as multi-variate mutual information, which we also use.

> We connect invariance properties of the representation to the geometry of the optimization residual, and to the phenomenon of flat minima (Dinh et al., 2017).

> Following (McAllester, 2013), we have also explored relations between our theory and the PAC-Bayes framework (Dziugaite and Roy, 2017). As we show, our theory can also be derived in the PAC-Bayes framework, without resorting to information quantities and the Information Bottleneck, thus providing both an independent and alternative derivation, and a theoretically rigorous way to upper-bound the optimal loss function. The use of PACBayes theory to study the generalization properties of deep networks has been championed by Dziugaite and Roy (2017), who point out that minima that are flat in the sense of having a large volume, toward which stochastic gradient descent algorithms are implicitly or explicitly biased (Chaudhari and Soatto, 2018), naturally relates to the PAC-Bayes loss for the choice of a normal prior and posterior on the weights. This has been leveraged by Dziugaite and Roy (2017) to compute non-vacuous PAC-Bayes error bounds, even for deep networks.

### $\mathbf{2.\;Preliminaries}$

> A training set $D=(x,y)$, where $x=x_{i=1}^{N}$ and $y=y_{i=1}^{N}$, is a collection of N randomly sampled data points $x_{i}^{i}$ and their associated (usually discrete) labels. The samples are assumed to come from an unknown, possibly complex, distribution $p_{\theta}(x,y)$, parametrized by a parameter $\theta$. Following a Bayesian approach, we also consider $\theta$ to be a random variable, sampled from some unknown prior distribution $p(\theta)$, but this requirement is not necessary (see Section 6). A test datum $x$ is also a random variable. Given a test sample, our goal is to infer the random variable $y$  which is therefore referred to as our task.

> We will make frequent use of the following standard information theoretic quantities (Cover and Thomas, 2012): Shannon entropy $H(x)=E_{p}[−\log{p(x)}]$, conditional entropy $H(x\vert{}y):=E_{\hat{y}}[H(x\vert{}y=\hat{y})]=H(x,y)-H(y)$, (conditional) mutual information $I(x;y\vert{}z)=H(x\vert{}z)-H(x\vert{}y,z)$, Kullbach-Leibler (KL) divergence $KL(p(x)\vert{}\vert{}q(x))=E_{p}[\log{}p/q]$, crossentropy H_{p,q}(x)=E_{p}[−\log{}q(x)], and total correlation $TC(z)$, which is also known as multi-variate mutual information and defined as

$$TC(z)=KL(p(z)\vert{}\vert{}\prod_{i}p(z_{i})),$$

> where $p(z_{i})$ are the marginal distributions of the components of $z$. Recall that the KL divergence between two distributions is always non-negative and zero if and only if they are equal. In particular $TC(z)$ is zero if and only if the components of $z$ are independent, in which case we say that $z$ is disentangled. We often use of the following identity:

$$I(z;x)=E_{x\sim{p(x)}}KL(p(z\vert{}x)\vert{}\vert{}p(z)).$$

> We say that $x$, $z$, $y$ form a Markov chain, indicated with $x\to{}z\to{}y$, if $p(y\vert{}x,z)=p(y\vert{}z)$. The Data Processing Inequality (DPI) for a Markov chain $x\to{}z\to{}y$ ensures that $I(x;z)\geq{}I(x;y): If $z$ is a (deterministic or stochastic) function of $x$, it cannot contain more information about $y$ than $x$ itself (we cannot create new information by simply applying a function to the data we already have).

#### $\mathbf{2.1\;General\;definitions\;and\;the\;Information\;Bottleneck\;Lagrangian}$

> We say that $z$ is a representation of $x$ if $z$ is a stochastic function of $x$, or equivalently if the distribution of $z$ is fully described by the conditional $p(z\vert{}x)$. In particular we have the Markov chain $y\to{}x\to{}z$. We say that a representation $z$ of $x$ is sufficient for $y$ if $y\perp{}x\vert{}z$, or equivalently if $I(z;y)=I(x;y)$; it is minimal when $I(x;z)$ is smallest among sufficient representations. To study the trade-off between sufficiency and minimality, Tishby et al. (1999) introduces the Information Bottleneck Lagrangian

$$L(p(z\vert{}x))=H(y\vert{}z)+\beta{}I(z;x),$$

> where $\beta$ trades off sufficiency (first term) and minimality (second term); in the limit $\beta\to{}0$, the IB Lagrangian is minimized when $z$ is minimal and sufficient. It does not impose any restriction on disentanglement nor invariance, which we introduce next.

#### $\mathbf{2.2\;Nuisances\;for\;a\;task}$

> A nuisance is any random variable that affects the observed data $x$, but is not informative to the task we are trying to solve. More formally, a random variable $n$ is a nuisance for the task $y$ if $y\perp{}n$, or equivalently $I(y;n)=0$. Similarly, we say that the representation $z$ is invariant to the nuisance $n$ if $z\perp{}n$, or $I(z;n)=0$. When $z$ is not strictly invariant but it minimizes $I(z;n)$ among all sufficient representations, we say that the representation $z$ is **maximally insensitive** to $n$.

> One typical example of nuisance is a group $G$. such as translation or rotation, acting on the data. In this case, a deterministic representation $f$ is invariant to the nuisances if and only if for all $g\in{}G$ we have $f(g·x)=f(x)$. Our definition however is more general in that it is not restricted to deterministic functions, nor to group nuisances. An important consequence of this generality is that the observed data $x$ can always be written as a deterministic function of the task $y$ and of all nuisances $n$ affecting the data, as explained by the following proposition.

> **Proposition 2.1 (Task-nuisance decomposition, Appendix C.1)** Given a joint distribution $p(x,y)$, where $y$ is a discrete random variable, we can always find a random variable $n$ independent of $y$ such that $x=f(y,n)$, for some deterministic function $f$.

### $\mathbf{3.\;Properties\;of\;optimal\;representations}$

> To simplify the inference process, instead of working directly with the observed high dimensional data $x$, we want to use a representation $z$ that captures and exposes only the information relevant for the task $y$. Ideally, such a representation should be (a) sufficient for the task $y$, i.e. $I(y;z)=I(y;x)$, so that information about $y$ is not lost; among all sufficient representations, it should be (b) minimal, i.e. $I(z;x)$ is minimized, so that it retains as little about $x$ as possible, simplifying the role of the classifier; finally, it should be (c) invariant to the effect of nuisances $I(z;n)=0$, so that the final classifier will not overfit to spurious correlations present in the training dataset between nuisances $n$ and labels $y$. Such a representation, if it exists, would not be unique, since any bijective mapping preserves all these properties. We can use this to our advantage and further aim to make the representation (d) maximally disentangled, i.e., choose the one(s) for which $TC(z)$ is minimal. This simplifies the classifier rule, since no information will be present in the higher-order correlations between the components of $z$. 

> Inferring a representation that satisfies all these properties may seem daunting. However,  in this section we show that we only need to enforce (a) sufficiency and (b) minimality, from which invariance and disentanglement follow naturally thanks to the stacking of noisy layers of computation in deep networks. We will then show that sufficiency and minimality of the learned representation can also be promoted easily through implicit or explicit regularization during the training process.

> **Proposition 3.1 (Invariance and minimality, Appendix C.2)** Let $n$ be a nuisance for the task $y$ and let $z$ be a sufficient representation of the input $x$. Suppose that $z$ depends on $n$ only through $x$ ( i.e., $n\to{}x\to{}z$). Then,

$$I(z;n)\leq{}I(z;x)-I(x;y).$$

> Moreover, there is a nuisance $n$ such that equality holds up to a (generally small) residual $\epsilon$

$$I(z;n)=I(z;x)-I(x;y)-\epsilon{}$$

> where $\epsilon{}:=I(z;y\vert{}n)-I(x;y)$. In particular $0\leq{}\epsilon{}\leq{}H(y\vert{}x)$, and $\epsilon=0$ whenever $y$ is a deterministic function of $x$. Under these conditions, a sufficient statistic $z$ is invariant (maximally insensitive) to nuisances if and only if it is minimal. 

> **Remark 3.2 Since** $\epsilon{}\leq{}H(y\vert{}x)$, and usually $H(y\vert{}x)=0$ or at least $H(y\vert{}x)\ll{}I(x;z)$, we can generally ignore the extra term.

> An important consequence of this proposition is that we can construct invariants by simply reducing the amount of information $z$ contains about $x$, while retaining the minimum amount $I(z;x)$ that we need for the task $y$. This provides the network a way to automatically learn invariance to complex nuisances, which is complementary to the invariance imposed by the architecture. Specifically, one way of enforcing minimality explicitly, and hence invariance, is through the IB Lagrangian. 

> **Corollary 3.3 (Invariants from the Information Bottleneck)** Minimizing the IB Lagrangian

$$L(p(z\vert{}x))=H(y\vert{}z)+\beta{}I(z;x),$$

> in the limit \beta\to{}0, yields a sufficient invariant representation $z$ of the test datum $x$ for the task $y$.

> Remarkably, the IB Lagrangian can be seen as the standard cross-entropy loss, plus a regularizer $I(z;x)$ that promotes invariance. This fact, without proof, is implicitly used in Achille and Soatto (2018), who also provide an efficient algorithm to perform the optimization. Alemi et al. (2017a) also propose a related algorithm and empirically show improved resistance to adversarial nuisances. In addition to modifying the cost function, invariance can also be fostered by choice of architecture:

> **Corollary 3.4 (Bottlenecks promote invariance)** Suppose we have the Markov chain of layers

$$x\to{}z_{1}\to{}z_{2}$$

> and suppose that there is a communication or computation bottleneck between $z_{1}$ and $z_{2}$ such that $I(z_{1};z_{2})<I(z_{1};x)$. Then, if $z_{2}$ is still sufficient, it is more invariant to nuisances than $z_{1}$. More precisely, for all nuisances $n$ we have $I(z_{2};n)\leq{}I(z_{1};z_{2})-I(x;y)$. Such a bottleneck can happen for example because $\dim(z_{2})<\dim(z_{1})$, e.g., after a pooling layer, or because the channel between $z_{1}$ and $z_{2}$ is noisy, e.g., because of dropout.

> **Proposition 3.5** (Stacking increases invariance) Assume that we have the Markov chain of layers

$$x\to{z_{1}}\to{}z_{2}\to{}\cdots{}\to{}z_{L},$$

> and that the last layer $z{L}$ is sufficient of $x$ for $y$. Then $z{L}$ is more insensitive to nuisances than all the preceding layers.

> Notice, however, that the above corollary does not simply imply that the more layers the merrier, as it assumes that one has successfully trained the network ($z{L}$ is sufficient), which becomes increasingly difficult as the size grows. Also note that in some architectures, such as ResNets (He et al., 2016), the layers do not necessarily form a Markov chain because of skip connections; however, their “blocks” still do.

> Proposition 3.6 (Actionable Information) When $z=f(x)$ is a deterministic invariant, if it minimizes the IB Lagrangian it also maximizes Actionable Information (Soatto, 2013), which is $H(x):=H(f(x))$.

> Although Soatto (2013) addressed maximal invariants, we only consider sufficient invariants, a advocated by (Soatto and Chiuso, 2016).

#### $\mathbf{Information\;in\;the\;weights}$

> Thus far we have discussed properties of representations in generality, regardless of how they are implemented or learned. Given a source of data (for example randomly generated, or from a fixed dataset), and given a (stochastic) training algorithm, the output weight $w$ of the training process can be thought as a random variable (that depends on the stochasticity of the initialization, training steps and of the data). We can therefore talk about the information that the weights contain about the dataset $D$ and the training procedure, which we denote by $I(w;D)$.

> Two extreme cases consist of the trivial settings where we use the weights to memorize he dataset (the most extreme form of overfitting), or where the weights are constant, or pure noise (sampled from a process that is independent of the data). In between, the amount of information the weights contain about the training turns out to be an important quantity both in training deep networks, as well as in establishing properties of the resulting representation, as we discuss in the next section.

> Note that in general we do not need to compute and optimize the quantity of information in the weights. Instead, we show that we can control it, for instance by injecting noise in the weights, drawn from a chosen distribution, in an amount that can be modulated between zero (thus in theory allowing full information about the training set to be stored in the weights) to an amount large enough that no information is left. We will leverage this property in the next sections to perform regularization.

### $\mathbf{4.\;Learning\;minimal\;weights}$

> In this section, we let $p_{\theta}(x,y)$ be an (unknown) distribution from which we randomly sample a dataset $D$. The parameter $\theta$ of the distribution is also assumed to be a random variable with an (unknown) prior distribution $p(\theta)$. For example $p_{\theta}$ can be a fairly general generative model for natural images, and $\theta$ can be the parameters of the model that generated our dataset. We then consider a deep neural network that implements a map $x\to{}fw(x):=q(·\vert{}x,w)$ from an input $x$ to a class distribution $q(y\vert{}x,w)$.1 In full generality, and following a Bayesian approach, we let the weights $w$ of the network be sampled from a parametrized distribution $q(w\vert{}D)$,whose parameters are optimized during training.2 The network is then trained in order to minimize the expected cross-entropy loss 3

$$H_{p,q}(y\vert{}x,w)=E_{D=(x,y)}E_{W\sim{}(w\vert{}D)}\sum_{i=1}^{N}-\log{}q(y^{i}\vert{}x^{i},w),$$

> in order for $q(y\vert{}x,w)$ to approximate $p_{\theta}(y\vert{}x)$.

> One of the main problems in optimizing a DNN is that the cross-entropy loss in notoriously prone to overfitting. In fact, one can easily minimize it even for completely random labels (see Zhang et al. (2017), and Figure 1). The fact that, somehow, such highly over-parametrized functions manage to generalize when trained on real labels has puzzled theoreticians and prompted some to wonder whether this may be inconsistent with the intuitive interpretation of the bias-variance trade-off theorem, whereby unregularized complex models should overfit wildly. However, as we show next, there is no inconsistency if one measures complexity by the information content, and not the dimensionality, of the weights.

> To gain some insights about the possible causes of over-fitting, we can use the following decomposition of the cross-entropy loss (we refer to Appendix C for the proof and the precise definition of each term):

$$H_{p,q}(y\vert{}x,w)=\underbrace{H(y\vert{}x,\theta{})}_{intrinsic\;error}+\underbrace{I(\theta{};y\vert{}x,w)}_{sufficiency}+E_{x,w}\underbrace{KL(p(y\vert{}x,w)\vert{}\vert{}q(y\vert{}x,w))}_{efficiency}-\underbrace{I(y:w\vert{}x,\theta{})}_{overfitting}$$

> The first term of the right-hand side of (8) relates to the intrinsic error that we would commit in predicting the labels even if we knew the underlying data distribution $p_{\theta}$; the second term measures how much information that the dataset has about the parameter $\theta$ is captured by the weights, the third term relates to the efficiency of the model and the class of functions fw with respect to which the loss is optimized. The last, and only negative, term relates to how much information about the labels, but uninformative of the underlying data distribution, is memorized in the weights. Unfortunately, without implicit or explicit regularization, the network can minimize the cross-entropy loss (LHS), by just maximizing the last term of eq. (8), i.e., by memorizing the dataset, which yields poor generalization.

> To prevent the network from doing this, we can neutralize the effect of the negative term by adding it back to the loss function, leading to a regularized loss $L=H_{p},q(y\vert{}x,w)+I(y;w\vert{}x,\theta)$. However, computing, or even approximating, the value of $I(y, w\vert{}x,\theta)$ is at least as difficult as fitting the model itself. 

> We can, however, add an upper bound to $I(y;w\vert{}x,\theta)$ to obtain the desired result. In particular, we explore two alternate paths that lead to equivalent conclusions under different premises and assumptions: In one case, we use a PAC-Bayes upper-bound, which is $KL(q(w\vert{}D) k p(w))$ where $p(w)$ is an arbitrary prior. In the other, we use the IB Lagrangian and upper-bound it with the information in the weights $I(w; D)$. We discuss this latter approach now, and look at the PAC-Bayes approach in Section 6.

> Notice that to successfully learn the distribution $p_{\theta}$, we only need to memorize in $w$ the information about the latent parameters $\theta$, that is we need $I(D;w)=I(D;\theta)\leq{}H(\theta)$, which is bounded above by a constant. On the other hand, to overfit, the term $I(y;w\vert{}x,\theta)\leq{}I(D; w\vert{}\theta)$ needs to grow linearly with the number of training samples $N$. We can exploit this fact to prevent overfitting by adding a Lagrange multiplier $\beta$ to make the amount of information a constant with respect to $N$, leading to the regularized loss function

$$L(q(w\vert{}D))=H_{p,q}(y\vert{}x,w)+\beta{}I(w;D),$$

> which, remarkably, has the same general form of an IB Lagrangian, and in particular is similar to (1), but now interpreted as a function of the weights $w$ rather than the activations $z$. This use of the IB Lagrangian is, to the best of our knowledge, novel, as the role of the Information Bottleneck has thus far been confined to characterizing the activations of the network, and not as a learning criterion. Equation (3) can be seen as a generalization of other suggestions in the literature:

> **IB Lagrangian, Variational Learning and Dropout.** Minimizing the information stored at the weights $I(w; D)$ was proposed as far back as Hinton and Van Camp (1993) as a way of simplifying neural networks, but no efficient algorithm to perform the optimization was known at the time. For the particular choice $\beta=1$, the IB Lagrangian reduces to the variational lower-bound (VLBO) of the marginal log-likelihood $p(y\vert{}x)$. Therefore, minimizing eq. (3) can also be seen as a generalization of variational learning. A particular case of this was studied by Kingma et al. (2015), who first showed that a generalization of Dropout, called Variational Dropout, could be used in conjunction with the reparametrization trick Kingma and Welling (2014) to minimize the loss efficiently.

> **Information in the weights as a measure of complexity.** Just as Hinton and Van Camp (1993) suggested, we also advocate using the information regularizer $I(w; D)$ as a measure of the effective complexity of a network, rather than the number of parameters dim(w), which is merely an upper bound on the complexity. As we show in experiments, this allows us to recover a version of the bias-variance trade-off where networks with lower information complexity underfit the data, and networks with higher complexity overfit. In contrast, there is no clear relationship between number of parameters and overfitting (Zhang et al., 2017). Moreover, for random labels the information complexity allows us to precisely predict the overfitting and underfitting behavior of the network (Section 7)

#### $\mathbf{4.1\;Computable\;upper-bound\;to\;the\;loss}$

> Unfortunately, computing $I(w,D)=E_{D}KL(q(w\vert{}D)KL(q(w\vert{}D)\vert{}\vert{}q(w))$ is still too complicated, since it requires us to know the marginal $q(w)$ over all possible datasets and trainings of the network. To avoid computing this term, we can use the more general upper-bound

$$E_{D}KL(q(w\vert{}D)\vert{}\vert{}q(w))\leq{}E_{D}KL(q(w\vert{}D)\vert{}\vert{}q(w))+KL(q(w)\vert{}\vert{}p(w))=E_{D}KL(q(w\vert{}D)\vert{}\vert{}p(w)),$$

> where $p(w)$ is any fixed distribution of the weights. Once we instantiate the training set, we have a single sample of $D$, so the expectation over $D$ becomes trivial. This gives us the following upper bound to the optimal loss function

$$L(q(w\vert{}D))=H_{p,q}(y\vert{}x,w)+\beta{}KL(q(w\vert{}D)\vert{}\vert{}p(w))$$

> Generally, we want to pick $p(w)$ in order to give the sharpest upper-bound, and to be a fully factorized distribution, i.e., a distribution with independent components, in order to make the computation of the KL term easier. The sharpest upper-bound to $KL(q(w\vert{}D)\vert{}\vert{}q(w))$ that can be obtained using a factorized distribution $p$ is obtained when $p(w):=\tilde{q}(w)=\prod_{i}q(w_{i})$ where $q(w_{i})$ denotes the marginal distributions of the components of $q(w)$. Notice that. once a training procedure is fixed, this may be approximated by training multiple times and approximating each marginal weight distribution. With this choice of prior, our final loss function becomes

$$L(q(w\vert{}D))=H_{p,q}(y\vert{}x,w)+\beta{}KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w))$$

> for some fixed distribution $\tilde{q}$ that approximates the real marginal distribution $q(w)$. The IB Lagrangian for the weights in eq. (3) can be seen as a generally intractable special case of eq. (5) that gives the sharpest upper-bound to our desired loss in this family of losses.

> In the following, to keep the notation uncluttered, we will denote our upper bound $KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w))$ to the mutual information $I(w;D)$ simply by $\tilde{I}(w;D)$, where

$$\tilde{I}(w;D):=KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w)=KL(q(w\vert{}D)\vert{}\vert{}\prod_{i}q(w_{i})).$$


#### $\mathbf{4.2\;Bounding\;the\;information\;in\;the\;weights\;of\;a\;network}$

> To derive precise and empirically verifiable statements about $\tilde{I}(w; D)$, we need a setting where this can be expressed analytically and optimized efficiently on standard architectures. To this end, following Kingma et al. (2015), we make the following modeling choices.

> **Modeling assumptions.** Let $w$ denote the vector containing all the parameters (weights) in the network, and let Wk denote the weight matrix at layer k. We assume an improper\loguniform prior on w, that is $\tilde{q}(wi)=c/\vert{}wi\vert{}$. Notice that this is the only scale-invariant prior (Kingma et al., 2015), and closely matches the real marginal distributions of the weights in a trained network (Achille and Soatto, 2018); we parametrize the weight distribution $q(w_{i}\vert{}D)$ during training as

$$w_{i}\vert{}D\sim{}\epsilon_{i}\hat{w}_{i},$$

> where ˆwi is a learned mean, and $\epsilon_{i}\sim{}\log{}N(−\alpha{}i/2,\alpha{}_{i})$ is i.i.d. multiplicative log-normal noise with mean 1 and variance $\exp(\alpha{}+{i})$−1.4 Note that while Kingma et al. (2015) uses this arametrization as a local approximation of the Bayesian posterior for a given (log-uniform) prior, we rather define the distribution of the weights $w$ after training on the dataset D to be $q(w\vert{}D)$.

> **Proposition 4.1 (Information in the weights, Theorem C.4)** Under the previous modeling assumptions, the upper-bound to the information that the weights contain about the dataset is

$$I(w;D)\leq\tilde{I}(w;D)=-\frac{1}{2}\sum_{i=1}^{\dim{(w)}}\log{}\alpha_{i}+C,$$

> where the constant C is arbitrary due to the improper prior.

> **Remark 4.2 (On the constant C)** To simplify the exposition, since the optimization is unaffected by any additive constant, in the following we abuse the notation and, under the modeling assumptions stated above, we rather define $\tilde{I}(w;D):=-\frac{1}{2}\sum_{i=1}^{\dim(w)}\log{}\alpha{}_{i}$. Neklyudov et al. (2017) also suggest a principled way of dealing with the arbitrary constant by using a proper\log-uniform prior.

> Note that computing and optimizing this upper-bound to the information in the weights is relatively simple and efficient using the reparametrization trick of Kingma et al. (2015).

#### $\mathbf{4.3\;Flat\;minima\;have\;low\;information}$

> Thus far we have suggested that adding the explicit information regularizer $I(w; D)$ prevents the network from memorizing the dataset and thus avoid overfitting, which we also confirm empirically in Section 7. However, real networks are not commonly trained with this regularizer, thus seemingly undermining the theory. However, even when not explicitly present, the term $I(w;D)$ is implicit in the use of SGD. In particular, Chaudhari and Soatto (2018) show that, under certain conditions, SGD introduces an entropic bias of a very similar form to the information in the weights described thus far, where the amount of information can be controlled by the learning rate and the size of mini-batches.

> Additional indirect empirical evidence is provided by the fact that some variants of SGD (Chaudhari et al., 2017) bias the optimization toward “flat minima”, that are local minima whose Hessian has mostly small eigenvalues. These minima can be interpreted exactly as having low information $I(w;D)$, as suggested early on by Hochreiter and Schmidhuber (1997): Intuitively, since the loss landscape is locally flat, the weights may be stored at lower precision without incurring in excessive inference error. As a consequence of previous claims, we can then see flat minima as having better generalization properties and, as we will see in Section 5, the associated representation of the data is more insensitive to nuisances and more disentangled. For completeness, here we derive a more precise relationship between flatness (measured by the nuclear norm of the loss Hessian), and the information content based on our model.

> **Proposition 4.3 (Flat minima have low information, Appendix C.5)** Let $\hat{w}$ be a local minimum of the cross-entropy loss $H_{p},q(y\vert{}x,w)$, and let H be the Hessian at that point. Then, for the optimal choice of the posterior $w\vert{}D=\epsilon{}\odot{}\hat{w}$ centered at wˆ that optimizes the IB Lagrangian, we have

$$I(w;D)\leq{}\tilde{I}(w;D)\leq{}\frac{1}{2}[\log{}\vert{}\vert{}\hat{w}\vert{}\vert{}_{2}^{2}+\log{}\vert{}\vert{}H\vert{}\vert{}_{*}-K\log{}(K^{2}\beta{}/2)]$$

> where K=dim(w) and k·k∗ denotes the nuclear norm.

> Notice that a converse inequality, that is, low information implies flatness, needs not hold, so there is no contradiction with the results of Dinh et al. (2017). Also note that for ˜I(w; D) to be invariant to reparametrization one has to consider the constant C, which we have ignored (Remark 4.2). The connection between flatness and overfitting has also been studied by Neyshabur et al. (2017), including the effect of the number of parameters in the model.

> In the next section, we prove one of our main results, that networks with low information in the weights realize invariant and disentangled representations. Therefore, invariance and disentanglement emerge naturally when training a network with implicit (SGD) or explicit (IB Lagrangian) regularization, and are related to flat minima.

### $\mathbf{5.\;Duality\;of\;the\;Bottleneck}$

> The following proposition gives the fundamental link in our model between information in the weights, and hence flatness of the local minima, minimality of the representation, and disentanglement.

> **Proposition 5.1 (Appendix C.6)** Let z=W $x$, and assume as before W=\epsilon{} Wˆ , with \epsiloni,j ∼\log{}N (−\alpha{}i/2, \alpha{}i). Further assume that the marginals of p(z) and $p(z\vert{}x)$ are both approximately Gaussian (which is reasonable for large dim(x) by the Central Limit Theorem). Then,

$$$$

> where Wi denotes the i-th row of the matrix W, and \alpha{}˜i is the noise variance \alpha{}˜i=exp(\alpha{}i)−1. In particular, I(z;x)+TC(z) is a monotone decreasing function of the weight variances \alpha{}i.

> The above identity is difficult to apply in practice, but with some additional hypotheses, we can derive a cleaner uniform tight bound on I(z;x)+TC(z).

> **Proposition 5.2 (Uniform bound for one layer, Appendix C.7)** Let z=W $x$, where W=\epsilonWˆ , where \epsiloni,j ∼\log{}N (−\alpha{}/2, \alpha{}); assume that the components of $x$ are uncorrelated, and that their kurtosis is uniformly bounded.5 Then, there is a strictly increasing function g(\alpha{}) s.t. we have the uniform bound

$$$$

> where c=O(1/ dim(x))\leq{}1, g(\alpha{})=−\log{}(1-e −\alpha{})/2 and \alpha{} is related to ˜I(w; D) by \alpha{}=exp {−I(W; D)/ dim(W)}. In particular, I(x;z)+T C(z) is tightly bounded by ˜I(W; D) and increases strictly with it. 

> The above theorems tells us that whenever we decrease the information in the weights, either by explicit regularization, or by implicit regularization (e.g., using SGD), we automatically improve the minimality, and hence, by Proposition 3.1, the invariance, and the disentanglement of the learner representation. In particular, we obtain as a corollary that SGD is biased toward learning invariant and disentangled representations of the data. Using the Markov property of the layers, we can easily extend this bound to multiple layers:

> **Corollary 5.3 (Multi-layer case, Appendix C.8)** Let Wk for k=1, ..., L be weight matrices, with Wk=\epsilon{}k  Wˆ k and \epsilon{}k i,j=log N (−\alpha{} k/2, \alpha{}k ), and let z_{i}+1=φ(Wk zk), where z0=x and φ is any nonlinearity. Then, 

$$$$

> where \alpha{} k=exp  −I(Wk ; D)/ dim(Wk ) . 

> **Remark 5.4 (Tightness)** While the bound in Proposition 5.2 is tight, the bound in the multilayer case needs not be. This is to be expected: Reducing the information in the weights creates a bottleneck, but we do not know how much information about $x$ will actually go through this bottleneck. Often, the final layers will let most of the information through, while initial layers will drop the most.

> **Remark 5.5 (Training-test transfer)** We note that we did not make any (explicit) assumption about the test set having the same distribution of the training set. Instead, we make the less restrictive assumption of sufficiency: If the test distribution is entirely different from the training one – one may not be able to achieve sufficiency. This prompts interesting questions about measuring the distance between tasks (as opposed to just distance between distributions), which will be studied in future work.

### $\mathbf{6.\;Connection\;with\;PAC-Bayes\;bounds}$

> In this section we show that using a PAC-Bayes bound, we arrive at the same regularized loss function eq. (5) we obtained using the Information Bottleneck, without the need of any approximation. By Theorem 2 of McAllester (2013), we have that for any fixed λ > 1/2, prior p(w), and any weight distribution q(w\vert{}D), the test error L test(q(w\vert{}D)) that the network commits using the weight distribution q(w\vert{}D) is upper-bounded in expectation by

$$$$

> where Lmax is the maximum per-sample loss function, which for a classification problem we can assume to be upper-bounded, for example by clipping the cross-entropy loss at chance level. Notice that right hand side coincides, modulo a multiplicative constant, with eq. (4) that we derived as an approximation of the IB Lagrangian for the weights (eq. (3)). 

> Now, recall that since we have 

$$$$

> the sharpest PAC-Bayes upper-bound to the test error is obtained when p(w)=q(w), in which case eq. (7) reduces (modulo a multiplicative constant) to the IB Lagrangian of the weights. That is, the IB Lagrangian for the weights can be considered as a special case of PAC-Bayes giving the sharpest bound.

> Unfortunately, as we noticed in Section 4, the joint marginal q(w) of the weights is not tractable. To circumvent the problem, we can instead consider that the sharpest PAC-Bayes upper-bound that can be obtained using a tractable factorized prior p(w), which is obtained exactly when p(w)=˜q(w)=Q i q(wi) is the product of the marginals, leading again to our practical loss eq. (5).

> On a last note, recall that under our modeling assumptions the marginal ˜q(w) is assumed to be an improper\log-uniform distribution. While this has the advantage of being a noninformative prior that closely matches the real marginal of the weights of the network, it also has the disadvantage that it is only defined modulo an additive constant, therefore making the bound on the test error vacuous under our model. 

> The PAC-Bayes bounds has also been used by Dziugaite and Roy (2017) to study the generalization property of deep neural networks and their connection with the optimization algorithm. They use a Gaussian prior and posterior, leading to a non-vacuous generalization bound.

### $\mathbf{7.\;Empirical\;validation}$

#### $\mathbf{7.1\;Transition\;from\;overfitting\;to\;underfitting}$

> As pointed out by Zhang et al. (2017), when a standard convolutional neural network (CNN) is trained on CIFAR-10 to fit random labels, the network is able to (over)fit them perfectly. This is easily explained in our framework: It means that the network is complex enough to memorize all the labels but, as we show here, it has to pay a steep price in terms of information complexity of the weights (Figure 2) in order to do so. On the other hand, when the information in the weights is bounded using and information regularizer, overfitting is prevented in a theoretically predictable way.

> In particular, in the case of completely random labels, we have I(y;w\vert{}x,\theta)=I(y;w)\leq{}I(w; D), where the first equality holds since $y$ is by construction random, and therefore independent of $x$ and $\theta$. In this case, the inequality used to derive eq. (3) is an equality, and the IBL is an optimal regularizer, and, regardless of the dataset size N, for \beta > 1 it should completely prevent memorization, while for \beta<1 overfitting is possible. To see this, notice that since the labels are random, to decrease the classification error by\log{}\vert{}Y\vert{}, where \vert{}Y\vert{} is the number of possible classes, we need to memorize a new label. But to do so, we need to store more information in the weights of the network, therefore increasing the second term I(w; D) by a corresponding quantity. This trade-off is always favorable when \beta<1, but it is not when \beta > 1. Therefore, the theoretically the optimal solution to eq. (1) is to memorize all the labels in the first case, and not memorize anything in the latter.

> As discussed, for real neural networks we cannot directly minimize eq. (1), and we need to use a computable upper bound to I(w; D) instead (Section 4.2). Even so, the empirical ehavior of the network, shown in Figure 1, closely follows this prediction, and for various sizes of the dataset clearly shows a phase transition between overfitting and underfitting near the critical value \beta=1. Notice instead that for real labels the situation is different:

![Figure 2]()

> Figure 2: (Left) Plot of the training error on CIFAR-10 with random labels as a function of the parameter $\beta$ for different models (see the appendix for details). As expected, all models show a sharp phase transition from complete overfitting to underfitting before the critical value \beta=1. (Right) We measure the quantity of information in the weights necessary to overfit as we vary the percentage of corrupted labels under the same settings of Figure 1. To fit increasingly random labels, the network needs to memorize more information in the weights; the increase needed to fit entirely random labels is about the same magnitude as the size of a label (2.30 nats/sample).

> The model is still able to overfit when \beta<1, but importantly there is a large interval of \beta > 1 where the model can fit the data without overfitting to it. Indeed, as soon as \betaN ∝ I(w; D) is larger than the constant H(\theta), the model trained on real data fits real labels without excessive overfitting (Figure 1). 

> Notice that, based on this reasoning, we expect the presence of a phase transition between an overfitting and an underfitting regime at the critical value \beta=1 to be largely independent on the network architecture: To verify this, we train different architectures on a subset of 10000 samples from CIFAR-10 with random labels. As we can see on the left plot of Figure 2, even very different architectures show a phase transition at a similar value of \beta. We also notice that in the experiment ResNets has a sharp transition close to the critical \beta.

> In the right plot of Figure 2 we measure the quantity information in the weights for different levels of corruption of the labels. To do this, we fix \beta<1 so that the network is able to overfit, and for various level of corruption we train until convergence, and then     ompute I(w; D) for the trained model. As expected, increasing the randomness of the labels increases the quantity of information we need to fit the dataset. For completely random labels, I(w; D) increases by ∼ 3 nats/sample, which the same order of magnitude as the quantity required to memorize a 10-class labels (2.30 nats/sample), as shown in Figure 2.

#### $\mathbf{7.2\;Bias-variance\;trade-off}$

> The Bias-Variance trade-off is sometimes informally stated as saying that low-complexity models tend to underfit the data, while excessively complex models may instead overfit, so that one should select an adequate intermediate complexity. This is apparently at odds with the common practice in Deep Learning, where increasing the depth or the number of weights of the network, and hence increasing the “complexity” of the model measured by the number of parameters, does not seem to induce overfitting. Consequently, a number of alternative measures of complexity have been proposed that capture the intuitive biasvariance trade-off curve, such as different norms of the weights (Neyshabur et al., 2015).

![Figure 3]()

> Figure 3: Plots of the test error obtained training the All-CNN architecture on CIFAR-10 (no data augmentation). (Left) Test error as we increase the number of weights in the network using weight decay but without any additional explicit regularization. Notice that increasing the number of weights the generalization error plateaus rather than increasing. (Right) Changing the value of \beta, which controls the amount of information in the weights, we obtain the characteristic curve of the bias-variance trade-off. This suggests that the quantity of information in the weights correlates well with generalization. 

> From the discussion above, we have seen that the quantity of information in the weights, or alternatively its computable upperbound ˜I(w; D), also provides a natural choice to measure model complexity in relation to overfitting. In particular, we have already seen that models need to store increasingly more information to fit increasingly random labels (Figure 2). In Figure 3 we show that by controlling ˜I(w; D), which can be done easily by modulating \beta, we recover the right trend for the bias-variance tradeoff, whereas models with too little information tend to underfit, while models memorizing too much information tend to overfit.

#### $\mathbf{7.3\;Nuisance\;invariance}$

> Corollary 5.3 shows that by decreasing the information in the weights I(w; D), which can be done for example using eq. (3), the learned representation will be increasingly minimal, and therefore insensitive to nuisance factors n, as measured by I(z;n). Here, we adapt a technique from the GAN literature Sønderby et al. (2017) that allows us to explicitly measure I(z;n) and validate this effect, provided we can sample from the nuisance distribution p(n) and from p(x\vert{}n); that is, if given a nuisance $n$ we can generate data $x$ affected by that nuisance. Recall that by definition we have

$$$$

> To approximate the expectations via sampling we need a way to approximate the likelihood ratio\log{}p(z\vert{}n)/p(z). This can be done as follows: Let D(z;n) be a binary discriminator that given the representation $z$ and the nuisance $n$ tries to decide whether $z$ is sampled from the posterior distribution p(z\vert{}n) or from the prior p(z). Since by hypothesis we can generate samples from both distributions, we can generate data to train this discriminator. Intuitively, if the discriminator is not able to classify, it means that $z$ is insensitive to changes of n. Precisely, since the optimal discriminator is 

$$$$

![Figure 4]()

> Figure 4: (Left) A few training samples generated adding nuisance clutter $n$ to the MNIST dataset. (Right) Reducing the information in the weights makes the representation $z$ learned by the digit classifier increasingly invariant to nuisances (I(n;z) decreases), while sufficiency is retained (I(z;y)=I(x;y) is constant). As expected, I(z;n) is smaller but has a similar behavior to the theoretical bound in Theorem 5.3. 

> if we assume that D is close to the optimal discriminator D∗ , we have

$$$$

> therefore we can use D to estimate the\log-likelihood ratio, and so also the mutual information I(z;n). Notice however that this comes with no guarantees on the quality of the approximation.

> To test this algorithm, we add random occlusion nuisances to MNIST digits (Figure 4). In this case, the nuisance $n$ is the occlusion pattern, while the observed data $x$ is the occluded digit. For various values of \beta, we train a classifier on this data in order to learn a representation $z$, and, for each representation obtained this way, we train a discriminator as described above and we compute the resulting approximation of I(z;n). The results in Figure 4 show that decreasing the information in the weights makes the representation increasingly more insensitive to n.

### $\mathbf{8.\;Discussion\;and\;conclusion}$

> In this work, we have presented bounds, some of which are tight, that connect the amount of information in the weights, the amount of information in the activations, the invariance property of the network, and the geometry of the residual loss. These results leverage the structure of deep networks, in particular the multiplicative action of the weights, and the Markov property of the layers. This leads to the surprising result that reducing information stored in the weights about the past (dataset) results in desirable properties of the learned internal representation of the test datum (future). 

> Our notion of representation is intrinsically stochastic. This simplifies the computation as well as the derivation of information-based relations. However, note that even if we start with a deterministic representation w, Proposition 4.3 gives us a way of converting it to a stochastic representation whose quality depends on the flatness of the minimum. Our theory uses, but does not depend on, the Information Bottleneck Principle, which dates back to over two decades ago, and can be re-derived in a different frameworks, for instance PAC-Bayes, which yield the same results and additional bounds on the test error. 

> This work focuses on the inference and learning of optimal representations, that seek to get the most out of the data we have for a specific task. This does not guarantee a good outcome since, due to the Data Processing Inequality, the representation can be easier to use but ultimately no more informative than the data themselves. An orthogonal but equally interesting issue is how to get the most informative data possible, which is the subject of active learning, experiment design, and perceptual exploration. Our work does not address transfer learning, where a representation trained to be optimal for a task is instead used for a different task, which will be subject of future investigations.

### $\mathbf{Acknowledgments}$

> Supported by ONR N00014-17-1-2072, ARO W911NF-17-1-0304, AFOSR FA9550-15-1-0229 and FA8650-11-1-7156. We wish to thank our reviewers and David McAllester, Kevin Murphy, Alessandro Chiuso for the many insightful comments and suggestions.

---

#### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> Alessandro Achille and Stefano Soatto. Information dropout: Learning optimal representations through noisy computation. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), PP(99):1–1, 2018.

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. In Proceedings of the International Conference on Learning Representations (ICLR), 2017a.

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy. Fixing a Broken ELBO. ArXiv e-prints, November 2017b.

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> Fabio Anselmi, Lorenzo Rosasco, and Tomaso Poggio. On invariance and selectivity in representation learning. Information and Inference, 5(2):134–158, 2016.

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> Zhaojun Bai, Gark Fahey, and Gene Golub. Some large-scale matrix computation problems. Journal of Computational and Applied Mathematics, 74(1-2):71–89, 1996.

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> Yoshua Bengio. Learning deep architectures for ai. Foundations and trends in Machine Learning, 2(1):1–127, 2009.

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> Sterling K. Berberian. Borel spaces, April 1988.

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> Joan Bruna and St´ephane Mallat. Classification with scattering operators. In IEEE Conference on Computer Vision and Pattern Recognition, pages 1561–1566, 2011.

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> Pratik Chaudhari and Stefano Soatto. Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks. Proc. of the International Conference on Learning Representations (ICLR), 2018.

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> Djork-Arn´e Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2012.

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008, 2017.

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning, volume 1. Springer series in statistics New York, 2001.

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> Ulf Grenander. General Pattern Theory. Oxford University Press, 1993.

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> Geoffrey E Hinton and Drew Van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the 6th annual conference on Computational learning theory, pages 5–13. ACM, 1993.

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> Sepp Hochreiter and J¨urgen Schmidhuber. Flat minima. Neural Computation, 9(1):1–42, 1997.

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR), 2014.

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> Diederik P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems 28, pages 2575–2583, 2015.

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> Yann LeCun. Learning invariant feature hierarchies. In Proceedings of the European Conference on Computer Vision (ECCV), pages 496–505, 2012.

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> Holden Lee, Rong Ge, Andrej Risteski, Tengyu Ma, and Sanjeev Arora. On the ability of neural nets to express distributions. In Proceedings of Machine Learning Research, volume 65, pages 1–26, 2017.

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> David McAllester. A pac-bayesian tutorial with a dropout bound. arXiv preprint arXiv:1307.2118, 2013.

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. In Proceedings of the 34 th International Conference on Machine Learning (ICML), 2017.

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> Kirill Neklyudov, Dmitry Molchanov, Arsenii Ashukha, and Dmitry P Vetrov. Structured bayesian pruning via log-normal multiplicative noise. In Advances in Neural Information Processing Systems 30, pages 6775–6784. Curran Associates, Inc., 2017.

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In Advances in Neural Information Processing Systems, pages 2422–2430, 2015.

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in deep learning. In Advances in Neural Information Processing Systems, pages 5949–5958, 2017.

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810, 2017.

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> Stefano Soatto. Actionable information in vision. In Machine learning for computer vision. Springer, 2013.

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> Stefano Soatto and Alessandro Chiuso. Visual representations: Defining properties and deep approximations. In Proceedings of the International Conference on Learning Representations (ICLR). 2016.

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Husz´ar. Amortised map inference for image super-resolution. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.

<a href="#footnote_36_2" name="footnote_36_1">[36]</a> Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.

<a href="#footnote_37_2" name="footnote_37_1">[37]</a> Ganesh Sundaramoorthi, Peter Petersen, V. S. Varadarajan, and Stefano Soatto. On the set of images modulo viewpoint and contrast changes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2009.

<a href="#footnote_38_2" name="footnote_38_1">[38]</a> Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In Information Theory Workshop (ITW), 2015 IEEE, pages 1–5. IEEE, 2015.

<a href="#footnote_39_2" name="footnote_39_1">[39]</a> Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. In The 37th annual Allerton Conference on Communication, Control, and Computing, pages 368–377, 1999.

<a href="#footnote_40_2" name="footnote_40_1">[40]</a> Greg Ver Steeg and Aram Galstyan. Maximally informative hierarchical representations of high-dimensional data. In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics, 2015.

<a href="#footnote_41_2" name="footnote_41_1">[41]</a> Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaoou Tang. From facial parts responses to face detection: A deep learning approach. In Proceedings of the IEEE International Conference on Computer Vision, pages 3676–3684, 2015.

<a href="#footnote_42_2" name="footnote_42_1">[42]</a> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.