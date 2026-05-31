.. MIMIQ Circuits documentation master file, created by
   sphinx-quickstart on Wed Aug  2 19:43:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

MIMIQ Circuits is a quantum computing framework and high-performance simulator developed by QPerfect.

.. grid:: 1 2 2 3
    :gutter: 3
    :padding: 0

    .. grid-item-card:: :octicon:`hourglass` Quick Start
        :link: quick_start
        :link-type: doc

        Get up and running with a simple example.

    .. grid-item-card:: :octicon:`download` Installation
        :link: manual/installation
        :link-type: doc

        Step-by-step installation guide.

    .. grid-item-card:: :octicon:`book` Manual
        :link: manual/index
        :link-type: doc

        Comprehensive guide to MimiqCircuits.

    .. grid-item-card:: :octicon:`beaker` Use Cases
        :link: usecases/index
        :link-type: doc

        Examples and tutorials (VQE, QFT, Grover, etc.).

    .. grid-item-card:: :octicon:`code` API Reference
        :link: apidocs/mimiqcircuits
        :link-type: doc

        Detailed API documentation.

    .. grid-item-card:: :octicon:`globe` QPerfect
        :link: https://qperfect.io
        :link-type: url

        Visit the QPerfect website.

    .. grid-item-card:: :octicon:`mark-github` GitHub
        :link: https://github.com/qperfect-io/mimiqcircuits-python
        :link-type: url

        View the source code on GitHub.


About MIMIQ Circuits
--------------------

MIMIQ Circuits allows you to develop and run your quantum algorithms beyond the limits of today's noisy intermediate scale quantum (NISQ) computers. 
It provides a seamless interface to **QPerfect's MIMIQ-CIRC**, a large-scale quantum circuit simulator capable of executing complex circuits with high fidelity.

**Quick Installation**

Get started immediately by installing the package via pip:

.. code-block:: bash

  pip install mimiqcircuits

Or optionally installing dependencies for the local simulator (quantanium) and visualization (matplotlib) tools:

.. code-block:: bash

  pip install mimiqcircuits[quantanium,visualization]

For detailed requirements and setup, see the :doc:`Installation <manual/installation>` page.

.. toctree::
   :hidden:

   mimiq_documentation
   quick_start
   manual/index
   usecases/index   
   API References <apidocs/mimiqcircuits>
   GitHub <https://github.com/qperfect-io/mimiqcircuits-python>
   Julia APIs <https://mimiq.qperfect.io/docs/julia/>
   QPerfect <https://qperfect.io>
