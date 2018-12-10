Tutorial `~adaptive.AverageLearner`s
------------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.AverageLearner`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()

`~adaptive.AverageLearner` (0D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next type of learner averages a function until the uncertainty in
the average meets some condition.

This is useful for sampling a random variable. The function passed to
the learner must formally take a single parameter, which should be used
like a “seed” for the (pseudo-) random variable (although in the current
implementation the seed parameter can be ignored by the function).

.. jupyter-execute::

    def g(n):
        import random
        from time import sleep
        sleep(random.random() / 1000)
        # Properly save and restore the RNG state
        state = random.getstate()
        random.seed(n)
        val = random.gauss(0.5, 1)
        random.setstate(state)
        return val

.. jupyter-execute::

    learner = adaptive.AverageLearner(g, atol=None, rtol=0.01)
    # `loss < 1` means that we reached the `rtol` or `atol`
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 1)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    runner.live_plot(update_interval=0.1)

`~adaptive.AverageLearner1D` and `~adaptive.AverageLearner2D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This learner is a combination between the `~adaptive.Learner1D` (or `~adaptive.Learner2D`)
and the `~adaptive.AverageLearner`, in a way such that it handles
stochastic functions with two variables.

Here, when chosing points the learner can either
* add more values to existing points
* add more intervals (or triangles)

So, the ``learner`` compares the loss of __potential new intervals (or triangles)__ with the __standard error__ of an existing point.

The relative importance of both can be adjusted by a hyperparameter ``learner.weight``, see the doc-string for more information.

Let's again try to learn some functions but now with uniformly distributed noise. We start with 1D and then go to 2D.

`~adaptive.AverageLearner1D`
............................

.. jupyter-execute::

    def noisy_peak(x_seed):
        import random
        x, seed = x_seed
        random.seed(x_seed)  # to make the random function deterministic
        a = 0.01
        peak = x + a**2 / (a**2 + x**2)
        noise = random.uniform(-0.5, 0.5)
        return peak + noise

    learner = adaptive.AverageLearner1D(noisy_peak, bounds=(-1, 1), weight=10)
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
    runner.live_info()

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    %%opts Image {+axiswise} [colorbar=True]
    # We plot the average

    def plotter(learner):
        plot = learner.plot()
        number_of_points = learner.mean_values_per_point()
        title = f'loss={learner.loss():.3f}, mean_npoints={number_of_points}'
        return plot.opts(plot=dict(title_format=title))

    runner.live_plot(update_interval=0.1, plotter=plotter)

`~adaptive.AverageLearner2D`
............................

.. jupyter-execute::

    def noisy_ring(xy_seed):
        import numpy as np
        import random
        (x, y), seed = xy_seed
        random.seed(xy_seed)  # to make the random function deterministic
        a = 0.2
        ring = x + np.exp(-(x**2 + y**2 - 0.75**2)**2/a**4)
        noise = random.uniform(-0.5, 0.5)
        return ring + noise

    learner = adaptive.AverageLearner2D(noisy_ring, bounds=[(-1, 1), (-1, 1)])
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
    runner.live_info()

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

See the average number of values per point with:

.. jupyter-execute::

    learner.mean_values_per_point()

Let's plot the average and the number of values per point.
Because the noise is uniform we expect the number of values per
point to be uniform too.

.. jupyter-execute::

    %%opts Image {+axiswise} [colorbar=True]
    # We plot the average

    def plotter(learner):
        plot = learner.plot()
        number_of_points = learner.mean_values_per_point()
        title = f'loss={learner.loss():.3f}, mean_npoints={number_of_points}'
        return plot.opts(plot=dict(title_format=title))

    runner.live_plot(update_interval=0.1, plotter=plotter)
