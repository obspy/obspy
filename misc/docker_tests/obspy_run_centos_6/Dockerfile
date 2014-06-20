FROM obspy:dependencies_centos_6

MAINTAINER Lion Krischer

ADD _install_and_run_tests_on_image.sh _install_and_run_tests_on_image.sh
ADD obspy /obspy
CMD ["/bin/bash", "_install_and_run_tests_on_image.sh"]
