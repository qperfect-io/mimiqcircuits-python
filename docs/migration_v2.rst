Migrating to MIMIQ V2
=====================

What is MIMIQ V2?
-----------------

MIMIQ V2 is the next generation of QPerfect's cloud service.
It runs the same MIMIQ simulators, so your code, circuits, and results work as before. 
Only the connection changes: V2 has its own server URL and 
logs you in through a new authentication service.

Among the benefits of this new generation is a 
refreshed web UI, 
better management of organizations, 
proper usage reports,
improved user support,
shorter queue times,
and, quicker feature rollout.

You can access the Frontend for MIMIQ cloud V2 at https://mimiq.qperfect.io/v2.



Getting ahead of the curve: enable V2 right now!
------------------------------------------------

First you will have to have an account on MIMIQ V2.
Current MIMIQ and the new version have separate databases and user accounts.
Please contact our team at mimiq.support@qperfect.io to request 
the creation of a new account or to move your existing account to the new platform.

Then, you will just have to pass the ``QPERFECT_CLOUD_V2`` URL when you create the connection.
The constant is available from :mod:`mimiqcircuits`.

.. code-block:: python

    from mimiqcircuits import MimiqConnection, QPERFECT_CLOUD_V2

    conn = MimiqConnection(QPERFECT_CLOUD_V2)
    conn.connect()

    # The remaining of your MIMIQ code stays as is

``connect()`` works the same way as with the current service. With no
arguments it opens a login page in your browser, and it also accepts a saved
refresh token or an email and password (see :ref:`connection methods <connection-methods>`).

The rest of your code (circuit defintions, submission, result collection, etc) stays as it is.

.. important::

    MIMIQ V2 will become the default after a few trial months.
    Once that happens, ``MimiqConnection()`` with no URL will connect to V2.
    We will be sure to give plenty of notice when a date is defined.


.. tip::

    To disable the warning when connecting to the current MIMIQ platform,
    set the ``MIMIQ_V2_WARNING`` environment variable to ``false`` or ``off``.

    - from a Linux shell, run ``export MIMIQ_V2_WARNING=off``
    - from a Jupyter notebook, run a cell with ``%env MIMIQ_V2_WARNING=off``
    - from a Python script, use ``os.environ["MIMIQ_V2_WARNING"] = "off"``


Requirements
------------

MIMIQ V2 needs ``mimiqcircuits`` 0.31.2 or later:

.. code-block:: bash

    pip install --upgrade mimiqcircuits


Existing data does not move to MIMIQ cloud V2
---------------------------------------------

The current MIMIQ cloud database is not integrated with the MIMIQ cloud V2 database.
That means that running simulations on one, won't show on the other.
We also won't move data between the two when V2 fully replaces the current system.
If needed, we can export data from the current cloud system for requesting users.
