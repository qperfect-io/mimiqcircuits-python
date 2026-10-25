Installation
============

In this section, we’ll walk you through the steps needed to get **MIMIQ Circuits** up and running on your 
system. We’ll cover everything from creating a virtual environment, installing the necessary dependencies, 
and installing the MIMIQ Circuits package, to setting up Jupyter for an enhanced coding experience. 
By the end of this guide, you’ll be ready to dive into quantum computing with MIMIQ Circuits.


Prerequisites
-------------

MIMIQ Circuits is compatible with **Python 3.10 or later**. To get started, ensure Python is installed on your system.  
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

Pre-built wheels are published for Linux (x86_64 and aarch64), macOS (x86_64
and arm64) and Windows (x86_64). On other platforms, and whenever pip installs
from source (for example from a git checkout), it compiles the package's Rust
extension. This needs a Rust toolchain, version 1.85 or newer, which you can
install from `rustup.rs <https://rustup.rs>`_.

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

To simulate circuits on your own machine, install Exaqt, QPerfect's local statevector simulator. It is free and is published on PyPI as a separate package:

.. code-block:: shell

    pip3 install mimiq-exaqt

See :doc:`local_statevector` for how to run circuits with it.

Jupyter Kernel Setup
--------------------

If you're using Jupyter, you need to install the Jupyter kernel for your virtual environment. Run the following command:

.. code-block:: shell

    python3 -m ipykernel install --user --name=<name-of-your-environment>

This will allow you to use the virtual environment within Jupyter by selecting the corresponding kernel from the **Kernel** menu.

Conclusion
----------

With these steps completed, you are now ready to explore the features and capabilities of MIMIQ Circuits within your Python environment. Happy coding!

For Developers
--------------

If you want to contribute to MIMIQ Circuits, we recommend using `uv <https://github.com/astral-sh/uv>`_ for dependency management.

**1. Install uv**

Follow the instructions on the `uv website <https://github.com/astral-sh/uv>`_ or run:

.. code-block:: shell

    curl -LsSf https://astral.sh/uv/install.sh | sh

**2. Install Rust**

``uv sync`` compiles the package's Rust extension, so a Rust toolchain (1.85 or
newer) is required. Install it from `rustup.rs <https://rustup.rs>`_ or run:

.. code-block:: shell

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

**3. Clone and Setup**

.. code-block:: shell

    git clone https://github.com/qperfect-io/mimiqcircuits-python.git
    cd mimiqcircuits-python
    uv sync

**4. Run Tests**

.. code-block:: shell

    uv run pytest

**5. Build Documentation**

.. code-block:: shell

    uv run --group docs --extra visualization make -C docs html
