import { Field } from "formik"

export default function BooleanField({ id, label, name }) {
    return (
      <label>
        <Field type = "checkbox" id = { id } name = { name } label = { label } />
        { label }
      </label>
    )
}