{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
   :members:

   .. comment to end block

{% if methods or attributes %}
   .. rubric:: Methods & Attributes
{% endif %}