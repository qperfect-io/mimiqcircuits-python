Installation
============

In this section, we’ll walk you through the steps needed to get **MIMIQ Circuits** up and running on your 
system. We’ll cover everything from creating a virtual environment, installing the necessary dependencies, 
and installing the MIMIQ Circuits package, to setting up Jupyter for an enhanced coding experience. 
By the end of this guide, you’ll be ready to dive into quantum computing with MIMIQ Circuits.


Prerequisites
-------------

MIMIQ Circuits is compatible with **Python 3.9 or later**. To get started, ensure Python is installed on your system.  
Download Python from the `official website <https://wiki.python.org/moin/BeginnersGuide/Download>`_.

For an interactive coding experience, we recommend using **Jupyter**.  
Install Jupyter by following the instructions at `Jupyter Installation <https://jupyter.org/install>`_.

To avoid dependency conflicts, we also recommend using Python virtual environments.  
Learn more about virtual environments from the `Python virtual environment guide <https://docs.python.org/3.10/tutorial/venv.html>`_.

Installing MIMIQ Circuits
-------------------------

Since MIMIQ Circuits is in rapid development, we recommend installing the latest version of MIMIQ Circuits Python directly from the GitHub repository.  
Ensure **git** is installed on your system by following the `Git installation guide <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

Step 1: Create a Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: Unix (Linux / macOS)

    .. code-block:: shell

        python3 -m venv /path/to/virtual/environment

.. tab:: Windows

    .. code-block:: powershell

        python3 -m venv c:\path\to\virtual\environment

Step 2: Activate the Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: Unix (Linux / macOS)

    .. code-block:: shell

        source /path/to/virtual/environment/bin/activate

.. tab:: Windows

    .. code-block:: powershell

        .\c:\path\to\virtual\environment\Scripts\Activate.ps1

Step 3: Install MIMIQ Circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use `pip` to install the latest version of MIMIQ Circuits from GitHub:

.. code-block:: shell

    pip3 install mimiqcircuits

Step 4: Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List the installed packages to confirm MIMIQ Circuits is installed correctly:

.. code-block:: shell

    pip3 list

Installing Optional Features
----------------------------

To use visualization functionality or work with Jupyter notebooks, install additional visualization support:

.. code-block:: shell

    pip3 install mimiqcircuits[visualization]

To use the local statevector simulator Quantanium:

.. code-block:: shell

    pip3 install mimiqcircuits[quantanium]

Jupyter Kernel Setup
--------------------

If you're using Jupyter, you need to install the Jupyter kernel for your virtual environment. Run the following command:

.. code-block:: shell

    python3 -m ipykernel install --user --name=<name-of-your-environment>

This will allow you to use the virtual environment within Jupyter by selecting the corresponding kernel from the **Kernel** menu.

Conclusion
----------

With these steps completed, you are now ready to explore the features and capabilities of MIMIQ Circuits within your Python environment. Happy coding!
