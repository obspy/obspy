{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
   :show-inheritance:

{% block attributes %}
{% if attributes %}
.. rubric:: Attributes

.. autosummary::
  :toctree: .
  :nosignatures:
{% for item in attributes %}
  ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock -%}
{% block methods %}
{%- if all_methods -%}
{% set public_methods = [] %}
{% set private_methods = [] %}
{% set special_methods = [] %}
{%- for m in all_methods -%}
{%- if m.startswith('_') and not m.startswith('__') -%}
{{ private_methods.append(m) or pass}}
{%- elif m.startswith('__') -%}
{{ special_methods.append(m) or pass}}
{%- else -%}
{{ public_methods.append(m) or pass}}
{%- endif -%}
{%- endfor -%}

{%- if public_methods -%}
.. rubric:: Public Methods

.. autosummary::
  :toctree: .
  :nosignatures:

{% for item in public_methods %}
  ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}

{% if private_methods %}
.. rubric:: Private Methods

.. warning::

  Private methods are mainly for internal/developer use and their API might change without notice.

{% for item in private_methods %}

.. automethod:: {{ name }}.{{ item }}


{%- endfor %}
{% endif %}
{% if special_methods %}
.. rubric:: Special Methods

{% for item in special_methods %}

.. automethod:: {{ name }}.{{ item }}

{%- endfor %}
{% endif %}
{% endif %}
{% endblock %}
