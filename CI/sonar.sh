#!/bin/bash

SONAR_SCANNER_VERSION="3.2.0.1227"
SONAR_ZIP_URL='https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-'${SONAR_SCANNER_VERSION}'-linux.zip';

function install_sonar {
    curl --insecure -o ./sonarscanner.zip -L $SONAR_ZIP_URL &&
    unzip sonarscanner.zip &&
    rm sonarscanner.zip &&
    mv sonar-scanner-${SONAR_SCANNER_VERSION}-linux sonar-scanner &&
    return 0
}

install_sonar;
if [ "$?" -ne "0" ]; then
    echo "Failure to install sonar";
    exit 1;
fi

version_number=$(head -n 1 "version.txt")

export PATH=$PATH:$PWD/sonar-scanner/bin
sonar-scanner -Dsonar.branch.name=$CI_COMMIT_REF_NAME -Dsonar.projectVersion=version_number

exit 0;
