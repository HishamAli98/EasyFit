Tutorials
=========

+++++++++++++++++++
EasyRegressor
+++++++++++++++++++


.. code-block:: python
    :caption: Train default models in _DEFAULT_REGRESSORS
    :emphasize-lines: 12
    :linenos:

    import sklearn

    from easyfit.regressors import EasyRegressor

    # Load dataset
    diabetes_dataset = sklearn.datasets.load_diabetes(as_frame=True)
    diabetes_df = diabetes_dataset['frame']
    X = diabetes_df.drop('target', axis=1)
    y = diabetes_df['target']

    # Create and train model
    model = EasyRegressor()
    model.fit(X, y)

    # Print the scores of the models
    print(model.score(X, y))


.. code-block:: python
    :caption: Train default models in _DEFAULT_REGRESSORS with additional models
    :emphasize-lines: 6
    :linenos:
    :lineno-start: 10

    # Create additional models_dict
    models_dict = {
        "additonal_model" : sklearn.linear_model.LinearRegression()
    }
    # Create and train model
    model = EasyRegressor(models_dict=models_dict, include_defaults=True)
    model.fit(X, y)

    # Print the scores of the models
    print(model.score(X, y))

.. code-block:: python
    :caption: Train only additional models
    :emphasize-lines: 6
    :linenos:
    :lineno-start: 10

    # Create additional models_dict
    models_dict = {
        "additonal_model" : sklearn.linear_model.LinearRegression()
    }
    # Create and train model
    model = EasyRegressor(models_dict=models_dict, include_defaults=False)
    model.fit(X, y)

    # Print the scores of the models
    print(model.score(X, y))


+++++++++++++++++++
EasyClassifier
+++++++++++++++++++


.. code-block:: python
    :caption: Train default models in _DEFAULT_CLASSIFIERS
    :emphasize-lines: 12
    :linenos:

    import sklearn

    from easyfit.classifiers import EasyClassifier

    # Load dataset
    diabetes_dataset = sklearn.datasets.load_iris(as_frame=True)
    diabetes_df = diabetes_dataset['frame']
    X = diabetes_df.drop('target', axis=1)
    y = diabetes_df['target']

    # Create and train model
    model = EasyClassifier()
    model.fit(X, y)

    # Print the scores of the models
    print(model.score(X, y))


.. code-block:: python
    :caption: Train default models in _DEFAULT_CLASSIFIERS with additional models
    :emphasize-lines: 6
    :linenos:
    :lineno-start: 10

    # Create additional models_dict
    models_dict = {
        "additonal_model" : sklearn.linear_model.LogisticRegression()
    }
    # Create and train model
    model = EasyClassifier(models_dict=models_dict, include_defaults=True)
    model.fit(X, y)

    # Print the scores of the models
    print(model.score(X, y))

.. code-block:: python
    :caption: Train only additional models
    :emphasize-lines: 6
    :linenos:
    :lineno-start: 10

    # Create additional models_dict
    models_dict = {
        "additonal_model" : sklearn.linear_model.LogisticRegression()
    }
    # Create and train model
    model = EasyClassifier(models_dict=models_dict, include_defaults=False)
    model.fit(X, y)

    # Print the scores of the models
    print(model.score(X, y))

    

