for model_name in '*' '*'
do
    echo '_trained_models/AS_basic/models'$model_name
    echo '_exports/AS_basic/exports'$model_name
    echo _logs/AS_basic/$model_name.log
    rm -rf _trained_models/AS_basic/models$model_name
    rm -rf _exports/AS_basic/exports$model_name
    rm -rf _logs/AS_basic/$model_name.log
    echo delete over
done
<<COMMENT
COMMENT