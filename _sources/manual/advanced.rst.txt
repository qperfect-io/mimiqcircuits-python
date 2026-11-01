Advanced Topics
===============

Take your quantum programming to the next level with advanced features. Learn how to interoperate with other frameworks, optimize your circuits, and utilize special data structures.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`file-code` Import & Export
        :link: import_export
        :link-type: doc

        Work with standard formats like **OpenQASM** and **Stim**, or use Mimiq's internal Protobuf format.

    .. grid-item-card:: :octicon:`rocket` Compilation
        :link: compilation
        :link-type: doc

        Optimize your circuits for execution. Learn about `remove_unused`, `remove_swaps` and other compilation passes.

    .. grid-item-card:: :octicon:`light-bulb` Special Topics
        :link: special_topics
        :link-type: doc

        Deep dive into specialized features like the :class:`~mimiqcircuits.BitString` class for classical register manipulation.

    .. grid-item-card:: :octicon:`plug` Implementing Backends
        :link: implementing_backends
        :link-type: doc

        Write a custom simulator that plugs into MIMIQ's unified backend API. Step-by-step guide with examples for local and remote backends, plus a conformance checklist.


.. toctree::
    :maxdepth: 2
    :hidden:

    import_export
    special_topics
    compilation
    implementing_backends
