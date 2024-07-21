############
Installation
############

.. _installation:

Installation
============

Let's get started using MIMIQ Circuits Python!

MIMIQ Circuits is compatible with Python 3.9 or later versions. To get started,
ensure that you have Python installed on your local system. You can download
Python from `Python Download <https://wiki.python.org/moin/BeginnersGuide/Download>`__.

For a smoother interaction, we recommend using Jupyter, an interactive
computing environment. Install Jupyter by following the instructions at
`Jupyter Installation <https://jupyter.org/install>`__.

To isolate MIMIQ Circuits and avoid conflicts, consider using Python virtual
environments. Create a minimal environment with only Python installed using
Python's built-in virtual environment tool. For guidance, refer to `Python
virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

Since MIMIQ Circuits is still in rapid development, we prefer to directly
install the latest available version directly from the `GitHub repository
<https://github.com/qperfect-io/mimiqcircuits-python.git>`__. In order to do so,
git needs to be installed. Please refer to the `Git Book
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__ for the
installation instructions.

To set up a minimal environment with only Python installed and begin using
MIMIQ Circuits, follow these steps:

1. Create a minimal environment

.. tab:: Unix (Linux / MacOS)

    .. code:: shell

        python3 -m venv /path/to/virtual/environment

.. tab:: Windows

    .. code:: powershell

        python3 -m venv c:\path\to\virtual\environment


2. Activate the newly created environment:

.. tab:: Unix (Linux / MacOS)

    .. code:: shell

        source /path/to/virtual/environment/bin/activate

.. tab:: Windows

    .. code:: powershell

        source c:\path\to\virtual\environment\bin\Activate.ps1


3. Install MIMIQ Circuits:

.. code:: shell

    pip3 install "mimiqcircuits @ git+https://github.com/qperfect-io/mimiqcircuits-python.git"


4. Verify the installed packages in your virtual environment:

.. code:: shell

    pip3 list


If you plan to use visualization functionality or work with Jupyter notebooks,
it is recommended to install the extra support for visualization:


.. code:: shell

    pip3 install "mimiqcircuits[visualization] @ git+https://github.com/qperfect-io/mimiqcircuits-python.git"


.. warning:: Jupyter Kernel

    If you are using Jupyter, you need to install the Jupyter kernel for your
    virtual environment. To do this, run the following command:

    .. code:: shell

        python3 -m ipykernel install --user --name=<name of your virtual environment>

    You can now use the virtual environment in Jupyter by selecting the
    corresponding kernel from the `Kernel` menu.

With these steps completed, you are now ready to explore the features and
capabilities of MIMIQ Circuits within your Python virtual environment. Happy
coding!
