{{ fullname }}
{{ underline }}

.. currentmodule:: {{ fullname }}
.. automodule:: {{ fullname }}

   .. comment to end block

   {% block functions %}

   {% set public_functions = [] %}
   {% set private_functions = [] %}
   {% for m in all_functions %}
   {% if m in functions %}
   {% do public_functions.append(m) %}
   {% else %}
   {% do private_functions.append(m) %}
   {% endif %}
   {%- endfor %}

   {% if public_functions %}
   .. rubric:: Public Functions

   .. autosummary::
      :toctree: .
      :nosignatures:

   {% for item in public_functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}

   {% if private_functions %}
   .. rubric:: Private Functions

   .. warning::

       Private functions are mainly for internal/developer use and their API might change without notice.

   .. autosummary::
     :toctree: .
     :nosignatures:
   {% for item in private_functions %}
       {{ item }}
   {%- endfor %}
   {% endif %}

   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree: .
      :nosignatures:

   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree: .
      :nosignatures:

   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
