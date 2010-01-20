{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. autosummary::
     :toctree: .
     :nosignatures:
   {% for item in methods %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
