import CodeEditor from "../ui/code-editor/code-editor"

import { Form } from "formik"

const CustomScript = ({}) => {

  return (
    <Form>
      <CodeEditor label = "Create Custom Script"/>
    </Form>
  )
}

export default CustomScript