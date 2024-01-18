namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: IYY19041_EXCPTN_MSG_FMT_AS_STR_S Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:49
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class IYY19041 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY19041_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    IYY19041_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY9041_IA Cyyy9041Ia;
    GEN.ORT.YYY.CYYY9041_OA Cyyy9041Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020308_esc_flag;
    //       +->   IYY19041_EXCPTN_MSG_FMT_AS_STR_S  01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_error iyy1_component (Transient,
    //       !                     Mandatory, Import only)
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           implementation_id
    //       !           specification_id
    //       !           dialect_cd
    //       !           activity_cd
    //       !       EXPORTS:
    //       !         Work View exp_error_msg iyy1_component (Transient,
    //       !                     Export only)
    //       !           severity_code
    //       !           message_tx
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  PURPOSE(CONTINUED)
    //     2 !   
    //     3 !  NOTE: 
    //     3 !  PRE-CONDITION
    //     3 !  
    //     3 !  POST-CONDITION
    //     3 !  
    //     3 !  See IRO10041
    //     3 !  
    //     4 !  NOTE: 
    //     4 !  RETURN / REASON CODES
    //     4 !  
    //     4 !  Bakınız IRO10041
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  RELEASE HISTORY
    //     5 !  01_00 23-02-1998 New release
    //     6 !   
    //     7 !  USE cyyy9041_excptn_msg_fmt_as_str
    //     7 !     WHICH IMPORTS: Work View imp_error iyy1_component TO Work
    //     7 !              View imp_error iyy1_component
    //     7 !     WHICH EXPORTS: Work View exp_error_msg iyy1_component FROM
    //     7 !              Work View exp_error_msg iyy1_component
    //     7 !                    Work View exp_error iyy1_component FROM Work
    //     7 !              View exp_error iyy1_component
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public IYY19041(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:49";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "IYY19041_EXCPTN_MSG_FMT_AS_STR_S";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	IYY19041_IA import_view, 
    	IYY19041_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      ++(NestingLevel);
      try {
        f_22020308_init(  );
        f_22020308(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020308(  )
    {
      func_0022020308_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020308" );
      Globdata.GetStateData().SetCurrentABName( "IYY19041_EXCPTN_MSG_FMT_AS_STR_S" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PURPOSE(CONTINUED)                                              
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    PRE-CONDITION                                                   
      //                                                                    
      //    POST-CONDITION                                                  
      //                                                                    
      //    See IRO10041                                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RETURN / REASON CODES                                           
      //                                                                    
      //    Bakınız IRO10041                                                
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      
      Cyyy9041Ia = (GEN.ORT.YYY.CYYY9041_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9041).Assembly,
      	"GEN.ORT.YYY.CYYY9041_IA" ));
      Cyyy9041Oa = (GEN.ORT.YYY.CYYY9041_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9041).Assembly,
      	"GEN.ORT.YYY.CYYY9041_OA" ));
      Cyyy9041Ia.ImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WIa.ImpErrorIyy1ComponentOriginServid);
      Cyyy9041Ia.ImpErrorIyy1ComponentContextString = StringAttr.ValueOf(WIa.ImpErrorIyy1ComponentContextString, 512);
      Cyyy9041Ia.ImpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WIa.ImpErrorIyy1ComponentReturnCode);
      Cyyy9041Ia.ImpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WIa.ImpErrorIyy1ComponentReasonCode);
      Cyyy9041Ia.ImpErrorIyy1ComponentImplementationId = DoubleAttr.ValueOf(WIa.ImpErrorIyy1ComponentImplementationId);
      Cyyy9041Ia.ImpErrorIyy1ComponentSpecificationId = DoubleAttr.ValueOf(WIa.ImpErrorIyy1ComponentSpecificationId);
      Cyyy9041Ia.ImpErrorIyy1ComponentDialectCd = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentDialectCd, 2);
      Cyyy9041Ia.ImpErrorIyy1ComponentActivityCd = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentActivityCd, 15);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9041).Assembly,
      	"GEN.ORT.YYY.CYYY9041",
      	"Execute",
      	Cyyy9041Ia,
      	Cyyy9041Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020308" );
      Globdata.GetStateData().SetCurrentABName( "IYY19041_EXCPTN_MSG_FMT_AS_STR_S" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentChecksum, 15);
      WOa.ExpErrorMsgIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorMsgIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorMsgIyy1ComponentMessageTx = StringAttr.ValueOf(Cyyy9041Oa.ExpErrorMsgIyy1ComponentMessageTx, 512);
      Cyyy9041Ia.FreeInstance(  );
      Cyyy9041Ia = null;
      Cyyy9041Oa.FreeInstance(  );
      Cyyy9041Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020308_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
      WOa.ExpErrorMsgIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorMsgIyy1ComponentMessageTx = "";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

