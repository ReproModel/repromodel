import { Field } from "formik"

export default function DefaultTextField({ id, label, name, type }) {
    return (
      <Field
        className = "inputField"
        id = { id }
        name = { name }
        label = { label }
        placeholder = { `Please enter ${type}` }
      />
    )
}