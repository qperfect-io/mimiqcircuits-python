MIMIQ Documentation
====================


Welcome to the documentation of MIMIQ, a quantum emulation platform developed by QPerfect.

Important Links
-------------------

-   `MIMIQ product page <https://qperfect.io/index.php/mimiq/>`_
-   :doc:`Install MIMIQ <manual/installation>`
-   :doc:`Quick start <quick_start>`
-   `Contact QPerfect <https://qperfect.io/#contact>`_


What is MIMIQ 
--------------------

MIMIQ is a feature-rich quantum computing framework that allows users to design quantum circuits and execute them on MIMIQ's remote service.

With MIMIQ you can test new quantum algorithms fast and at scale, emulate their implementation on noisy quantum hardware, and grow your understanding of quantum computing by peeking into the properties of the state during emulation. From variational quantum algorithms to quantum error correction, MIMIQ is a universal emulator well-equipped to handle most quantum circuits of interest.

MIMIQ's advantage
~~~~~~~~~~~~~~~~~~~~~~

| We developed MIMIQ as a tool for quantum computing researchers and application developers. 
| Therefore, it has been built for speed, scale and accuracy, as well as ease of use. We achieve this by providing:

- efficient implementations of state-of-the-art simulation and compression techniques, which allows to emulate small circuits fast, and large circuits (up to 100s-1000s of qubits) accurately and with fewer resources.

- a cloud interface that gives you greater power for large-scale emulation at your fingertips and allows to parallel execute multiple jobs without disturbing the workflow.

- simple interface and customizable functionality that facilitates the process of designing and executing quantum circuits.

- access to a professional set of advanced features for circuit composition (noise, conditional logic, mid-circuit measurements...), access to full quantum state properties (state amplitudes, expectation values, entanglement measures...), and workflow integration (parsers).

Simulation Methods
~~~~~~~~~~~~~~~~~~~~~~~~

| MIMIQ gives access to efficient implementations of state-of-the-art emulation methods.
| This includes a State Vector simulator for perfectly accurate simulations on small-scale circuits and a Matrix Product State simulator for larger-scale simulations with a controlled and transparent fidelity metric.
| To see more about the simulators available on the platform head to the :doc:`Simulators <manual/simulation>` page.

How to use the Documentation
--------------------------------

Exploring the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| On the left panel is located the tree of the different pages that can be found in this documentation. You can click on each part to access the pages of your interest.
| Alternatively if you need to look for something more specific (e.g. a specific function) you can use the search bar above the tree.
| From Python's interactive interpreter REPL you can also invoke the `help()` function and give it function (Class, Method, etc) you're interested in to get some information.

Documentation Content
~~~~~~~~~~~~~~~~~~~~~~~~~~

| If you are using MIMIQ for the first time it is highly recommended to head to the :doc:`installation <manual/installation>` and :doc:`quick start <quick_start>` pages to install MIMIQ and understand its basic functionnalities.  
| The core of documentation can be found in the "Manual" where every feature offered by MIMIQ will be covered with explanations and examples.
| If you encounter any issues you can contact us at `QPerfect <https://qperfect.io/#contact>`_.
| Finally for more in depth explanation of the different functions available you can head to the API section.

