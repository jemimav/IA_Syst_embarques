# import joblib
# import os

# def generate_linear_regression_code(coefs):
#     code = "#include <stdio.h>\n\nfloat prediction(float *features, int n_features) {\n"
#     code += "\tfloat coefs[%d] = {%f" % (len(coefs) + 1, coefs[0])
#     for i in range(1, len(coefs)):
#         code += ", %f" % coefs[i]
#     code += "};\n"
#     code += "\tfloat y = coefs[0];\n"
#     for i in range(1, len(coefs)):
#         code += "\ty += coefs[%d] * features[%d-1];\n" % (i, i)
#     code += "\treturn y;\n}\n"
#     return code


# def generate_logistic_regression_code(coefs):
#     code = "#include <stdio.h>\n#include <math.h>\n\n"
#     code += "float sigmoid(float x) {\n\treturn 1.0 / (1.0 + exp(-x));\n}\n\n"
#     code += "float prediction(float *features, int n_features) {\n"
#     code += "\tfloat coefs[%d] = {%f" % (len(coefs), coefs[0])
#     for i in range(1, len(coefs)):
#         code += ", %f" % coefs[i]
#     code += "};\n"
#     code += "\tfloat y = coefs[0];\n"
#     for i in range(1, len(coefs)):
#         code += "\ty += coefs[%d] * features[%d-1];\n" % (i, i)
#     code += "\treturn sigmoid(y);\n}\n"
#     return code

# def generate_decision_tree_code(tree):
#     code = "#include <stdio.h>\n\nfloat prediction(float *features, int n_features) {\n"
#     code += "\tfloat y = "
#     tree_str = tree.export_text()
#     tree_str = tree_str.replace('|', '\t')
#     tree_str = tree_str[:-2]
#     code += tree_str.replace('\n', '\n\t')
#     code += ";\n"
#     code += "\treturn y;\n}\n"
#     return code

# def transpile_model(model_name, filename):
#     model = joblib.load(filename)
#     coefs = None
#     tree = None
#     if hasattr(model, 'coef_'):
#         coefs = model.coef_
#     elif hasattr(model, 'tree_'):
#         tree = model.tree_
#     else:
#         raise ValueError("Unsupported model type: %s" % type(model))
#     if coefs is not None:
#         code = generate_linear_regression_code(coefs)
#     elif tree is not None:
#         code = generate_decision_tree_code(tree)
#     else:
#         raise ValueError("Unsupported model type: %s" % type(model))
#     with open('%s.c' % model_name, 'w') as f:
#         f.write(code)
#     os.system('gcc -o %s %s.c' % (model_name, model_name))
#     print("Compiled %s.c" % model_name)

# if __name__ == '__main__':
#     transpile_model("linear_regression", "linear_regression.joblib")
#     transpile_model("logistic_regression", "logistic_regression.joblib")
#     transpile_model("decision_tree", "decision_tree.joblib")

# import os
# import joblib
# from sklearn.tree import export_text


# def generate_linear_regression_code(theta):
#     code = "#include <stdio.h>\n\n"
#     code += "float prediction(float *features, int n_features) {\n"
#     code += "\tfloat sum = 0.0;\n"
#     code += "\tfloat x0 = 1.0;\n"
#     for i in range(len(theta)):
#         code += f"\tfloat x{i+1} = features[{i}];\n"
#         code += f"\tsum += {theta[i]} * x{i+1};\n"
#     code += "\treturn sum;\n}"
#     return code


# def generate_logistic_regression_code(theta):
#     code = "#include <stdio.h>\n#include <math.h>\n\n"
#     code += "float sigmoid(float x) {\n"
#     code += "\treturn 1.0 / (1.0 + exp(-x));\n}\n\n"
#     code += "float prediction(float *features, int n_features) {\n"
#     code += "\tfloat sum = 0.0;\n"
#     for i in range(len(theta)):
#         code += f"\tfloat x{i} = features[{i}];\n"
#         code += f"\tsum += {theta[i]} * x{i};\n"
#     code += "\tfloat z = sum;\n"
#     code += "\tfloat y = sigmoid(z);\n"
#     code += "\treturn y;\n}"
#     return code


# def generate_decision_tree_code(model):
#     n_features = model.tree_.n_features
#     feature_names = [f"X_{i+1}" for i in range(n_features)]

#     code = "#include <stdio.h>\n\n"
#     code += "int simple_tree(float *features, int n_features) {\n"
#     for i in range(n_features):
#         code += f"\tfloat x{i+1} = features[{i}];\n"
#     code += export_text(model, feature_names=feature_names).replace("|", "\t").replace("\n", "\n\t")[:-2]
#     code += "}\n\n"
#     code += "int main() {\n"
#     code += "\tfloat X[] = {1.0, -2.0};\n"  # You may need to update this line depending on the number of features
#     code += "\tint prediction = simple_tree(X, 2);\n"
#     code += "\tprintf(\"Prediction : %d\\n\", prediction);\n"
#     code += "\treturn 0;\n}"
#     return code



# def transpile_model(model_type, filename):
#     model = joblib.load(filename)
#     if model_type == "linear_regression":
#         code = generate_linear_regression_code(model.coef_)
#     elif model_type == "logistic_regression":
#         code = generate_logistic_regression_code(model.coef_[0])
#     elif model_type == "decision_tree":
#         code = generate_decision_tree_code(model)
#     else:
#         raise ValueError("Invalid model type")
#     with open(f"{model_type}.c", "w") as f:
#         f.write(code)
#     os.system(f"gcc -o {model_type} {model_type}.c")
#     print(f"Compiled {model_type}.c")


# if __name__ == '__main__':
#     transpile_model("linear_regression", "linear_regression.joblib")
#     transpile_model("logistic_regression", "logistic_regression.joblib")
#     transpile_model("decision_tree", "decision_tree.joblib")


# import sys
# import joblib

# def generate_c_code(model):
#     code = f"#include <stdio.h>\n\n"
#     code += f"float prediction(float *features, int n_features) {{\n"
#     code += f"    float result = {model.intercept_};\n"

#     for i, coef in enumerate(model.coef_):
#         code += f"    result += features[{i}] * {coef};\n"

#     code += f"    return result;\n"
#     code += f"}}\n\n"

#     code += f"int main() {{\n"
#     code += f"    float features[] = {{1, 1, 1}};\n"
#     code += f"    int n_features = sizeof(features) / sizeof(features[0]);\n\n"

#     code += f"    float pred = prediction(features, n_features);\n"
#     code += f"    printf(\"La prédiction pour les caractéristiques données est : %f\\n\", pred);\n\n"

#     code += f"    return 0;\n"
#     code += f"}}\n"

#     return code

# if __name__ == "__main__":
#     input_model_file = sys.argv[1]
#     output_c_file = sys.argv[2]

#     model = joblib.load(input_model_file)
#     c_code = generate_c_code(model)

#     with open(output_c_file, "w") as f:
#         f.write(c_code)

#     print(f"Le fichier C a été généré : {output_c_file}")
#     print(f"Pour le compiler, exécutez la commande : gcc {output_c_file} -o prediction -lm")

import sys
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression

def generate_c_code(model, model_type, output_file):
    if model_type == 'linear_regression':
        coefficients = model.coef_
        intercept = model.intercept_
        n_features = len(coefficients)

        # Convert the coefficients to a Python list
        coefficients_list = coefficients.tolist()

        c_code = f"""
#include <stdio.h>

float prediction(float *features, int n_features) {{
    float result = {intercept:.10f};
    float coefficients[] = {{{', '.join(map(str, coefficients))}}};

    for (int i = 0; i < n_features; i++) {{
            result += features[i] * coefficients[i];
        }}

        return result;
    }}

    int main() {{
        float test_features[] = {{1, 2, 3}};
        int n_features = {n_features};

        float predicted_value = prediction(test_features, n_features);
        printf("Prediction: %f\\n", predicted_value);

        return 0;
    }}
        """
    elif model_type == 'logistic_regression':
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        n_features = len(coefficients)

        # Convert the coefficients to a Python list
        coefficients_list = coefficients.tolist()

        c_code = f"""
#include <stdio.h>
#include <math.h>

float sigmoid(float z) {{
    return 1.0 / (1.0 + exp(-z));
}}

float prediction(float *features, int n_features) {{
    float z = {intercept:.10f};
    float coefficients[] = {{{', '.join(map(str, coefficients))}}};

    for (int i = 0; i < n_features; i++) {{
            z += features[i] * coefficients[i];
        }}

        return sigmoid(z);
    }}

    int main() {{
        float test_features[] = {{1, 2, 3}};
        int n_features = {n_features};

        float predicted_value = prediction(test_features, n_features);
        printf("Prediction: %f\\n", predicted_value);

        return 0;
    }}
        """
    elif model_type == 'decision_tree':
        n_features = model.tree_.n_features
        # Récupérez les informations de l'arbre de décision, comme les seuils, les caractéristiques, etc.
        # Pour cet exemple, nous utilisons un arbre de décision simple à deux niveaux.
        # Vous devez extraire les informations appropriées de l'objet "model" pour un arbre de décision réel.

        c_code = f"""
        #include <stdio.h>
        #include <stdbool.h>

        int simple_tree(float *features, int n_features) {{
            return (features[0] <= 0) && (features[1] <= 0);
        }}

        int main() {{
            float test_features[] = {{1, 2, 3}};
            int n_features = {n_features};

            int predicted_value = simple_tree(test_features, n_features);
            printf("Prediction: %d\\n", predicted_value);

            return 0;
        }}
        """
    else:
        print("Unsupported model type.")
        sys.exit(1)

    with open(output_file, "w") as f:
        f.write(c_code)

    print(f"Code C généré et sauvegardé dans {output_file}")
    print(f"Pour compiler, utilisez la commande suivante :")
    print(f"gcc -o {output_file[:-2]} {output_file} -lm")


if __name__ == "__main__":

    model_regression = joblib.load("linear_regression.joblib")
    model_logistic = joblib.load("logistic_regression.joblib")
    model_tree = joblib.load("decision_tree.joblib")
    

    generate_c_code(model_regression , 'linear_regression', "linear_regression.c")
    generate_c_code(model_logistic , 'logistic_regression', "logistic_regression.c")
    generate_c_code(model_tree , 'decision_tree', "decision_tree.c")
